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
<!-- TODO: consider requiring refine_to rather than refine, so that it's possible to make smart decisions about how to prioritize increasing the precision of different parts of the answer in order to obtain a desired precision-->

# features

- there is a function (`Computable::refine_to`) which takes a computable number $(x, b, f)$ and a precision $\epsilon$ and applies $f$ to $x$ until $b_u(f^n(x)) - b_\ell(f^n(x)) \leq \epsilon$ (where $n$ is the number of applications required).
- computable numbers may be composed using arithmetic operations via the standard operators (`+`, `-`, `*`, `/`, unary `-`) and `Computable::inv`.
for example, given computable numbers $C_0 = (x_0, b_0, f_0)$ and $C_1 = (x_1, b_1, f_1)$, $$C_0 + C_1 = ((x_0, x_1), (x, y) \mapsto (b_{0\ell}(x) + b_{1\ell}(y), b_{0u}(x) + b_{1u}(y)), (x, y) \mapsto (f_0(x), f_1(y)))$$

# deviations from the formalism

sadly, the implementation cannot exactly realize the formalism.

- many operations are fallible: bounds functions and composed operations return `Result` rather than only the types specified above. the refinement function $f$ itself is infallible in the implementation, but `Computable::refine_to` can fail when validating the refinement progress
- refinement is bounded: `Computable::refine_to` stops after a maximum number of iterations and returns an error instead of looping forever. note that default iteration limits differ by build: debug builds use a smaller max to catch issues quickly, while release builds allow more refinements for accuracy.
- we do not (and cannot. but maybe if we had an actual proof system...) enforce that the provided $f$ actually satisfies the convergence requirement from the formalism; this is the caller's responsibility. violations may lead to runtime errors. the implementation only checks that, on refinement, the state does change and the bounds don't get worse (since these are necessary conditions which are easy to check).

<!-- TODO: reconsider `Exponent = i64` vs `BigInt` for a more faithful D = Z × Z representation. -->

# internal design of computable numbers

## key features
- no recomputation: if the same computable number is used multiple times in an expression, when that expression is refined, the refinement of the computable number is shared between all instances
- hiding internal state: although the computable number will mutate its state $x$ on refinement, the users of the computable number can't see the state directly. they may only perceive it indirectly via time required to return (if long, the state must not have been refined much yet) and returned precision (if in excess of requested, the computable number was probably already refined to a greater precision than requested).
- parallelism: if an expression being refined has multiple components that need to be refined separately, those sub-refinements run in parallel.

## design
- i use the term 'composition' to refer to a computable number which contains multiple base computable numbers. for example, $\sqrt{a + ab}$ is a composition. $a + a$ is also considered a composition even though the constituent computable numbers are identical. (however, $2a$ is not a composition; it has only a single constituent to refine.)
- compositions are structured as binary trees; each composition may have at most two children. (note that the same computable number can occur multiple times in a single expression, so it's logically a DAG, but it's still structured as a binary tree) <!-- it's possible that this binary tree requirement will need to be relaxed, but i'm going to start out assuming that it does not -->
- when a composition is refined, all its branches are refined in parallel.
- refinements of the branches are propagated upwards live, and refinement is halted as soon as the overall expression reaches the required precision.
<!-- TODO: add a propagated stopping condition so branches can be halted when they can no longer tighten the overall bounds. -->

### example
let's consider the example of refining $\sqrt{a + ab}$ to precision $\epsilon=1$.
- the current value of the expression is computed
    - step inside the $\sqrt{}$
        - consider both sides of the addition in parallel
            - left branch: get the current bounds on $a$. suppose these are $(-1, 0.5)$
            - right branch
                - consider both sides of the multiplication in parallel
                    - get the current bounds on $a$: $(-1, 0.5)$
                    - get the current bounds on $b$: $(4, 6)$
                - combine both sides of the multiplication to get bounds on $ab$: $(-6, 3)$
        - combine both sides of the addition to get bounds on $a + ab$: $(-7, 3.5)$
    - apply $\sqrt{}$ to these bounds: recoverable error, since the bounds are not fully contained in the domain of $\sqrt{}$. refinement is required. (if the bounds did not contain any of the domain of $\sqrt{}$, we would finish with an irrecoverable error.)
- refine $\sqrt{a + ab}$
    - step inside the $\sqrt{}$
        - consider both sides of the addition in parallel
            - left branch: acquire the refinement lock on $a$ and refine $a$ repeatedly. on each refinement, publish the new bounds for reading elsewhere. let's suppose the successive refinements are $(-0.5, 0.5)$, $(-0.25, 0.25)$, $(0.125, 0.25)$, ...
            - right branch: consider both sides of the multiplication in parallel
                - left branch: try to acquire the refinement lock on $a$. since the refinement lock is already acquired, instead subscribe to updates on $a$'s bounds.
                - right branch: acquire the refinement lock on $b$. let's suppose refining $b$ is quite slow, and no refinements come in for now.
            - the multiplication receives refinements of its left branch $a$ (and is listening for refinements of $b$, but we're supposing refining $b$ is slow). when a new refinement comes in, it recalculates the multiplication bounds. for the successive refinements of $a$ that we supposed, this would yield $(-3, 3)$, $(-1.5, 1.5)$, $(0.75, 1.5)$, ...
        - similarly, the addition receives refinements of its left branch $a$ and right branch $ab$ and recalculates the addition bounds. the refinements from left and right may come in in any order; let's suppose this yields $(-6.5, 3.5)$, $(-3.5, 3.5)$, $(-2, 2)$, $(0.25, 2)$. (note that it is possible for the right expression $ab$ to be using a more refined $a$ than the left expression $a$, if the addition recieves the left update slowly.)
    - the $\sqrt{}$ receives refinements of its argument and recomputes itself. for $(-6.5, 3.5)$, $(-3.5, 3.5)$, and $(-2, 2)$ the bounds are still not fully contained in the domain of $\sqrt{}$, so it does not halt the refinement. for $(0.25, 2)$, since $\sqrt{}$ is increasing, the bounds are $(\sqrt{0.25}, \sqrt{2})$. these are both represented as computable numbers themselves, so to figure out if the desired precision has been attained, they too must be refined. both of these refinements proceed in parallel with each other and with $a$ and $b$ and the expressions built from them (including $\sqrt{}$ itself--it does not wait until the refinement of $\sqrt{0.25}$ and $\sqrt{2}$ is done to recompute).
        - refining $\sqrt{0.25}$: let's suppose that our initial bounds are $(0.25, 1)$ and our refinement algorithm is binary search (though in practice, hopefully we'd be using an algorithm that quickly detects that there is an exact answer). this yields $(0.25, 0.625)$, $(0.4375, 0.625)$, $(0.4375, 0.53125)$, ...
        - refining $\sqrt{2}$: let's suppose that our initial bounds are $(1, 2)$ and our refinement algorithm is binary search. this yields $(1, 1.5)$, $(1.25, 1.5)$, $(1.375, 1.5)$, $(1.375, 1.4375)$ ...
    - the $\sqrt{}$ receives the refinements of $\sqrt{0.25}$ and $\sqrt{2}$, also in some arbitrary order. it will compute the outer bounds to see if they become narrower than $\epsilon$ (meaning the precision has been attained), and the inner bounds to see if they become wider than our $\epsilon$ of 1 (meaning the desired precision cannot be obtained without further refining $a + ab$ itself). let's suppose that it recieves refinements of $\sqrt{2}$ and $\sqrt{0.25}$ in alternation. then it will obtain outer bounds of $(0.25, 2)$ and inner bounds of $(1, 1)$; $(0.25, 1.5)$ and $(1, 1)$; $(0.25, 1.5)$ and $(0.625, 1)$; $(0.25, 1.5)$ and $(0.625, 1.25)$; $(0.4375, 1.5)$ and $(0.625, 1.25)$; $(0.4375, 1.5)$ and $(0.625, 1.375)$; $(0.4375, 1.5)$ and $(0.53125, 1.375)$; $(0.4375, 1.4375)$ and $(0.53125, 1.375)$. the outer bounds now have a width of 1, which is exactly $\epsilon$, so refinement stops. the refinement of $\sqrt{a + ab}$ to $\epsilon=1$ returns $(0.4375, 1.4375)$, and all the other parallel processes are halted. (however, any refinements made to $a$ and $b$ which were already performed but didn't percolate up to $\sqrt{}$ in time to affect the output will be maintained, and if $\sqrt{a + ab}$ is refined again with a smaller $\epsilon$, they will be available at the start of that computation.)

(note that, in general, the constraints on a composition may be narrower than the combination of the constraints on each side considered independently; for example, $a - ab$ with bounds on $a$ of $(-1, 0.5)$ and bounds on $b$ of $(4, 6)$ is bounded by $(-2.5, 5)$, but considering $a$ and $ab$ independently (i.e. ignoring the fact that $a$ is in both) would yield $(-4, 6.5)$. we ignore this for now, always considering the sides independently, but this might be a place for further improvements.)


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

- everything is designed not to panic, and to instead return `Result`.
- we avoid interior mutability whenever possible.
