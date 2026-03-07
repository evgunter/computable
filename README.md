# goal

this project defines computable numbers in a way that is suitable for performing reasonably efficient operations with guarantees of correctness.

# formalism

we represent a computable number in the following way.

first, definitions:
- let $D = \mathbb{Z} \times \mathbb{Z}$ be the set of binary numbers: for binary number $(m, e)$, the number represented is $m \times 2^e$
- let $D_\infty = D \cup \{-\infty, +\infty\}$ be the set of extended binary bounds
- let $B = \{(x, y) \in D_\infty \times D_\infty \, | \, x \leq y\}$ be the set of valid bounds

then, a computable number is represented as an element of $(X, X \to B, X \to X)$, where:
- $x \in X$ is the 'current state': some information representing the status of the computation, which can be any type.
- a function $b : X \to B$ which calculates the bounds based on the current state. for convenience, we use $b_\ell$ and $b_u$ to denote the projections onto the lower and upper bounds. so, if $c \in \mathbb{R}$ is the true value of the computable number being represented, then $b_\ell \leq c \leq b_u$.
- a refinement function $f : X \to X$ which takes the state $x$ and returns a new state for which the calculated lower and upper bounds are no looser than before, i.e. $b_\ell(f(x)) \geq b_\ell(x) \text{ and } b_u(f(x)) \leq b_u(x)$. repeated refinement by $f$ must converge to a single value, i.e. $\lim_{n \to \infty} b_\ell(f^n(x)) = \lim_{n \to \infty} b_u(f^n(x))$.

for example, $\sqrt{2}$ could be represented as a computable number by setting $x$ to just be the current lower and upper bound (e.g. $x$ could be initialized as $(0, 2)$), with an $f$ which takes the midpoint between the lower and upper bounds, squares it, compares it to 2, and depending on the result replaces the upper or lower bound with the former midpoint.

there are a few ways the implementation diverges from this abstraction; see [deviations from the formalism](#deviations-from-the-formalism) below.

# features

- there is a function (`Computable::refine_to`) which takes a computable number $(x, b, f)$ and a precision $\epsilon$ and applies $f$ to $x$ until $b_u(f^n(x)) - b_\ell(f^n(x)) \leq \epsilon$ (where $n$ is the number of applications required).
- computable numbers may be composed using arithmetic operations via the standard operators (`+`, `-`, `*`, `/`, unary `-`) and `Computable::inv`.
for example, given computable numbers $C_0 = (x_0, b_0, f_0)$ and $C_1 = (x_1, b_1, f_1)$, $$C_0 + C_1 = ((x_0, x_1), (x, y) \mapsto (b_{0\ell}(x) + b_{1\ell}(y), b_{0u}(x) + b_{1u}(y)), (x, y) \mapsto (f_0(x), f_1(y)))$$

# deviations from the formalism

sadly, the implementation cannot exactly realize the formalism.

- many operations are fallible: bounds functions and composed operations return `Result` rather than only the types specified above. the refinement function $f$ itself is infallible in the implementation, but `Computable::refine_to` can fail when validating the refinement progress
- refinement is bounded: `Computable::refine_to` stops after a maximum number of iterations and returns an error instead of looping forever. note that default iteration limits differ by build: debug builds use a smaller max to catch issues quickly, while release builds allow more refinements for accuracy.
- we do not (and cannot. but maybe if we had an actual proof system...) enforce that the provided $f$ actually satisfies the convergence requirement from the formalism; this is the caller's responsibility. violations may lead to runtime errors. the implementation only checks that, on refinement, the state does change (if the width hasn't converged to zero) and the bounds don't get worse (since these are necessary conditions which are easy to check).
- the formalism above suggests that composed computables have a combined refinement function $(x, y) \mapsto (f_0(x), f_1(y))$ that directly calls both children's refinement functions. in the implementation, combinators (`AddOp`, `MulOp`, `NegOp`) are passive—they don't refine anything themselves. instead, `RefinementGraph` discovers all leaf "refiner" nodes (base computables and `InvOp`) and drives them externally, then propagates bounds upward through the passive combinators. this is functionally equivalent but architecturally different. <!-- TODO: consider whether to update the formalism to match the implementation, or vice versa -->


# internal design of computable numbers

## key features
- no recomputation: if the same computable number is used multiple times in an expression, when that expression is refined, the refinement of the computable number is shared between all instances
- hiding internal state: although the computable number will mutate its state $x$ on refinement, the users of the computable number can't see the state directly. they may only perceive it indirectly via time required to return (if long, the state must not have been refined much yet) and returned precision (if in excess of requested, the computable number was probably already refined to a greater precision than requested).
- parallelism: if an expression being refined has multiple components that need to be refined separately, those sub-refinements run in parallel.

## threading model
the implementation spawns one thread per refiner node (base computable numbers and active operations like `inv`) using `std::thread::scope`. threads are created per `refine_to` call and cleaned up when refinement completes. refiner threads are passive workers: they block waiting for commands from the coordinator and send back update messages.

## design
- i use the term 'composition' to refer to a computable number which contains multiple base computable numbers. for example, $\sqrt{a + ab}$ is a composition. $a + a$ is also considered a composition even though the constituent computable numbers are identical. (however, $2a$ is not a composition; it has only a single constituent to refine.)
- compositions are structured as binary trees; each composition may have at most two children. (note that the same computable number can occur multiple times in a single expression, so it's logically a DAG, but it's still structured as a binary tree)
- refinement is coordinated by `RefinementGraph`, which discovers all leaf "refiner" nodes and drives them externally. combinators (`AddOp`, `MulOp`, `NegOp`) are passive—they don't refine anything themselves. the coordinator sends `Step` commands to refiner threads and, after each response, propagates the updated bounds upward through the passive combinators.
- the refinement model is round-based with three key improvements over plain lock-step:
    - **early exit**: the coordinator checks whether root precision meets the target after each individual refiner response, not just at the end of a round. this means refinement can stop as soon as any single update tips the root bounds within tolerance.
    - **per-refiner exhaustion**: refiners that have converged (bounds are a single point) or whose state is unchanged are marked inactive and excluded from future rounds, rather than causing the entire refinement to fail.
    - **demand-based skipping**: each round computes a demand budget ($\epsilon / 2^{\lceil \log_2 N \rceil}$ where $N$ is the number of active refiners). refiners whose bounds are already narrower than the budget are skipped, avoiding wasted work on fast-converging operands. a safety valve ensures that if all active refiners are below the demand budget but root precision isn't met, the least-precise refiners are stepped anyway (skipping extreme outliers whose width is negligible compared to the widest).
<!-- TODO: add support for distributive law/commutativity-type-things, where e.g. Mul(Add(a, b), c) can be converted to Add(Mul(a, c), Mul(b, c)) or Mod_n(Mul(a, b)) can be converted to Mod_n(Mul(Mod_n(a), Mod_n(b))) -->

### example
let's consider the example of refining $\sqrt{a + ab}$ to precision $\epsilon=1$, where $a$ has initial bounds $(-1, 0.5)$ and $b$ has initial bounds $(4, 6)$.

the graph has three refiner nodes ($a$, $b$, and $\sqrt{}$) and two passive combinators ($+$ and $\times$). the coordinator spawns one thread per refiner and proceeds in rounds.

**round 1**: the coordinator computes the demand budget from $\epsilon$ and the number of active refiners, and sends a `Step` command to each refiner whose bounds are wider than the budget (initially, all of them).

- the refiner threads execute one refinement step each in parallel:
    - $a$ refines from $(-1, 0.5)$ to $(-0.5, 0.5)$
    - $b$ refines from $(4, 6)$ to $(4, 5)$
    - $\sqrt{}$'s refiner does its own internal step
- as each refiner responds, the coordinator propagates the update upward. for example, after $a$'s update arrives:
    - $a \times b$ is recomputed using $a$'s new bounds and $b$'s current bounds
    - $a + ab$ is recomputed
    - $\sqrt{a + ab}$ is recomputed
    - the coordinator checks whether the root bounds are within $\epsilon$; if so, refinement stops immediately (early exit)

**subsequent rounds**: the coordinator repeats—computing the demand budget, skipping refiners that are already precise enough, stepping the rest, and checking precision after each response. suppose after a few rounds:
- $b$ has converged to $(4.5, 4.5)$ and is marked exhausted (per-refiner exhaustion)—it won't be stepped again
- $a$'s bounds are now narrow enough relative to the demand budget and are skipped (demand-based skipping)
- only $\sqrt{}$'s refiner is stepped

this continues until the root bounds are within $\epsilon$, at which point all refiner threads are stopped and the result is returned.

(note that, in general, the constraints on a composition may be narrower than the combination of the constraints on each side considered independently; for example, $a - ab$ with bounds on $a$ of $(-1, 0.5)$ and bounds on $b$ of $(4, 6)$ is bounded by $(-2.5, 5)$, but considering $a$ and $ab$ independently (i.e. ignoring the fact that $a$ is in both) would yield $(-4, 6.5)$. we ignore this for now, always considering the sides independently, but this might be a place for further improvements.)

# design of binary numbers
- computable numbers are backed by `Binary` numbers, which are represented by $(m, e)$ where $m$ and $e$ are integers; the number represented is $m * 2^e$
- the integers themselves are represented with `BigInt`
- we also use "extended" binary numbers (`XBinary`), i.e. the above but also including $-\infty$ and $+\infty$
- it is also frequently useful to represent only nonnegative binary numbers; for this we have a separate type `UBinary` which is represented in the same way except that $m \geq 0$ (or internally, $m$ is a `BigUint`). similarly, there is a `UXBinary` type (which is used to represent the width of bounds.)

# norms

## usage norms

- all calculations on computable numbers should produce computable numbers as output
- bounds computation should be lightweight; expensive work should occur in refinement
- a computable number should only have its bounds refined when necessary, e.g. immediately before the final output or when required to satisfy conditions on the input of another function. for example:
    - a program which graphs a computable number to the screen should refine the computable number to some $\epsilon$ smaller than the pixel size right before graphing
    - a program which finds the (real) square root of a computable number and then does something with the result may refine that number until the lower bound is nonnegative (or the upper bound becomes negative and triggers an error). (it would also be permissible to defer the detection of this error until later, when the result is actually used.)
- to apply a function $g$ to computable numbers, you must define a function $G$ which computes a bound on $g$'s output based on bounds on its input such that the output bounds converge if the input bounds do.
Then $g((x, b, f)) = (x, G(b), f)$.
for example, if $g(y) = y^2$, its corresponding $G : B \to B$ has $(\ell, u) \mapsto (u^2, \ell^2)$ if $\ell,u \leq 0$, $(\ell, u) \mapsto (\ell^2, u^2)$ if $\ell,u \geq 0$, and $(\ell, u) \mapsto (0, \max(\ell^2, u^2))$ if $\ell \leq 0$ and $u \geq 0$. 
`Computable::inv` demonstrates this pattern.
<!-- TODO: add a general helper for applying bounded functions. -->

## design norms

- everything is designed not to panic, and to instead return `Result`
- we avoid interior mutability whenever possible
- whenever possible, we use the type system to constrain the inputs to belong to a correct type, rather than checking that the inputs are valid and returning a `Result` type
- for mathematically impossible cases that can't (yet) be prevented by the type system, use `unreachable!()` with a TODO about investigating type-system solutions. see `src/error.rs` for details.
