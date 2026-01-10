# goal

this project defines computable numbers in a way that is suitable for performing reasonably efficient operations with guarantees of correctness.

# formalism

we represent a computable number in the following way.

first, definitions:
- let $D = \mathbb{Z} \times \mathbb{Z}$ be the set of binary numbers: for binary number $(m, e)$, the number represented is $m \times 2^e$
- let $D_\infty = D \cup \{-\infty, +\infty\}$ be the set of extended binary bounds
- let $B = \{(x, y) \in D_\infty \times D_\infty \, | \, x \leq y\}$ be the set of valid bounds

then, a computable number is represented as an element of $(X, X \to B, X \times D \to X)$, where:
- $x \in X$ is the 'current state': some information representing the status of the computation, which can be any type.
- a function $b : X \to B$ which calculates the bounds based on the current state. for convenience, we use $b_\ell$ and $b_u$ to denote the projections onto the lower and upper bounds. so, if $c \in \mathbb{R}$ is the true value of the computable number being represented, then $b_\ell \leq c \leq b_u$.
- a refinement function $r : X \times D \to X$ which takes the state $x$ and a desired precision $\epsilon$ and returns a new state for which the bounds (a) are no looser than before and (b) have width at most $\epsilon$. repeated refinement to smaller $\epsilon$ must converge to a single value. (the purpose of the state is to avoid the need for recomputation if epsilon is progressively decreased.)

for example, $\sqrt{2}$ could be represented as a computable number by setting $x$ to just be the current lower and upper bound (e.g. $x$ could be initialized as $(0, 2)$), with an $r$ that repeatedly (a) takes the midpoint between the lower and upper bounds, (b) squares it, (c) compares it to 2, and (d) depending on the result replaces the upper or lower bound with the former midpoint, until the bounds are within $\epsilon$ of each other.

note that this definition does not represent *only* computable numbers--there's no guarantee that the upper and lower bounds actually converge. it's the responsibility of whatever constructs the computable number to ensure that repeated applications of $r$ actually make the bounds converge! (maybe if we had an actual proof system we could require a proof that it does converge...)

there are a few other ways the implementation diverges from this abstraction; see [deviations from the formalism](#deviations-from-the-formalism) below.

# features

- there is a function (`Computable::refine_to`) which takes a computable number $(x, b, r)$ and a precision $\epsilon$ and applies $r$ to $x$ until $b_u(x) - b_\ell(x) \leq \epsilon$, where `r` is a refinement procedure that may decide how to prioritize increasing the precision of composed parts.
- computable numbers may be composed using arithmetic operations via `Computable::add`, `Computable::sub`, `Computable::mul`, `Computable::div`, `Computable::neg`, and `Computable::inv`.
for example, given computable numbers $C_0 = (x_0, b_0, f_0)$ and $C_1 = (x_1, b_1, f_1)$, $$C_0 + C_1 = ((x_0, x_1), (x, y) \mapsto (b_{0\ell}(x) + b_{1\ell}(y), b_{0u}(x) + b_{1u}(y)), (x, y) \mapsto (f_0(x), f_1(y)))$$

# deviations from the formalism

sadly, the implementation cannot exactly realize the formalism.

- many operations are fallible: bounds computation, refinement, and composed operations return `Result` rather than just the types specified above.
- refinement is bounded: `Computable::refine_to` stops after a maximum number of iterations and returns an error instead of looping forever. note that default iteration limits differ by build: debug builds use a smaller max to catch issues quickly, while release builds allow more refinements for accuracy.
- we do not (and cannot) enforce that the provided refinement function actually satisfies the convergence requirement from the formalism; this is the caller's responsibility. violations may lead to runtime errors. the implementation only checks that, on refinement, the bounds don't get worse and the state changes.
<!-- TODO: reconsider `Exponent = i64` vs `BigInt` for a more faithful D = Z × Z representation. -->

# norms

## usage norms

- all calculations on computable numbers should produce computable numbers as output
- a computable number should only have its bounds refined when necessary, e.g. immediately before the final output or when required to satisfy conditions on the input of another function. for example:
    - a program which graphs a computable number to the screen should refine the computable number to some $\epsilon$ smaller than the pixel size right before graphing
    - a program which finds the (real) square root of a computable number and then does something with the result may refine that number until the lower bound is nonnegative (or the upper bound becomes negative and triggers an error). (it would also be permissible to defer the detection of this error until later, when the result is actually used.)
- to apply a function $g$ to computable numbers, you must define a function $G$ which computes a bound on $g$'s output based on bounds on its input such that the output bounds converge if the input bounds do.
Then $g((x, b, r)) = (x, G(b), r)$.
for example, if $g(y) = y^2$, its corresponding $G : B \to B$ has $(\ell, u) \mapsto (u^2, \ell^2)$ if $\ell,u \leq 0$, $(\ell, u) \mapsto (\ell^2, u^2)$ if $\ell,u \geq 0$, and $(\ell, u) \mapsto (0, \max(\ell^2, u^2))$ if $\ell \leq 0$ and $u \geq 0$. 
`Computable::inv` demonstrates this pattern.

## design norms

- everything is designed not to panic, and to instead return `Result`.
- we avoid interior mutability whenever possible.
