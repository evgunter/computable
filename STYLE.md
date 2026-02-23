# Rust Style Guide

This guide covers non-obvious conventions that go beyond what `rustfmt` and default `clippy` enforce. If something here seems obvious, it's included because it's easy to get wrong in practice.

## Naming

- **Iterator types match their producing method.** `keys()` returns `Keys`, `into_iter()` returns `IntoIter`. Never `KeyIterator`.
- **Prefer `to_`/`as_`/`into_` over `from_`** for named conversion methods — they chain better. (`From` trait impls are still encouraged for trait interop.)
- **Error types: verb-object-error order.** `ParseAddrError`, not `AddrParseError`.
- **Prefer full words over abbreviations.** `diagnostic` not `diag`, `transaction` not `txn`. Well-known domain abbreviations are acceptable when they're clearer to the team than the full form.
- **Don't shadow standard type names.** Don't define `type Result<T> = std::result::Result<T, MyError>` or similarly redefine `Option`, `Vec`, etc. Readers should be able to trust that standard names mean the standard things.

## Types & Data Structures

- **No trait bounds on struct definitions.** Write `struct Foo<T>` and put bounds on `impl<T: Clone> Foo<T>`. Bounds on the struct make every new derive a potential breaking change.
- **Enums with a natural null state get a `None` variant.** Don't wrap in `Option<Foo>` if `Foo` inherently has an absent/empty state. `Option<Foo>` means "sometimes nullable"; `Foo::None` means "always has a null state."
- **Never use default generic type parameters.** Be explicit; defaults hide complexity and make APIs harder to understand.
- **Public fields over getters.** Rust's module-level encapsulation makes OO-style getters unnecessary in most cases. Use getters only when the type has invariants that need protection.

## Traits

- **Design traits for object safety** when they might plausibly be used as trait objects, even if you only use generics now. Use `where Self: Sized` escape hatches for methods that can't be object-safe.
- **Use sealed traits** (public trait inheriting a private supertrait) when you want freedom to add methods without breaking downstream. Document why a trait is sealed.
- **Accept the widest type, produce the narrowest type** (variance). For example, `Fn` > `FnMut` > `FnOnce` for parameters (widest bound for callers), and reverse for implementations (most permissive trait for implementers).
- **Prefer `impl Trait` over newtypes for opaque return types**, unless you need to name the type to add trait impls. More generally, lean on traits and the trait system for abstraction — traits are Rust's primary abstraction mechanism and should be preferred over alternatives like enum dispatch or manual vtables.

## Functions

- **Smart pointer methods should be static.** `Box::into_raw(b)` not `b.into_raw()`. Inherent methods on smart pointers are ambiguous with `Deref`-coerced methods on the inner type.
- **Expose intermediate/byproduct results.** If a function computes useful byproduct data, return it. Like `Vec::binary_search` returning `Result<usize, usize>` (found index or insertion point), not just `bool`.
- **Accept `Read`/`Write` by value.** `fn foo(r: impl Read)` not `fn foo(r: &mut impl Read)`. `&mut R` itself implements `Read` when `R: Read`, so callers can pass `&mut reader` to retain ownership.
- **Avoid out-parameters.** Return values via tuples or structs. If you must use `&mut` out-parameters for performance (buffer reuse), the reason must be explicitly documented and backed by benchmark evidence.
- **Don't use `impl Into<T>` for implicit conversions** in function parameters. It hides type conversions without clear benefit and makes APIs harder to understand.

## Expressions & Control Flow

- **`let _ = x` drops immediately; `let _foo = x` lives to scope end.** `_` is not a binding — the value is discarded and its destructor runs immediately. `_foo` is a real binding that lives until end of scope. This matters for lock guards, file handles, and any RAII type.
- **Prefer functional style.** Use iterator combinators (`map`, `filter`, `fold`, `collect`) over imperative `for` loops. Functional chains are more composable, often more concise, and make data flow explicit. Use `for` loops only when closures would be large, when side effects dominate, or when the imperative version is genuinely clearer.

## Modules & Imports

- **Never import enum variants** into scope. Always write `MyEnum::Variant`. Exception: `Some`, `None`, `Ok`, `Err` from the prelude, and large enums in a local match scope.
- **Qualify imports when there's any possibility of confusion.** If a function or constant name is generic or ambiguous, import its parent module and qualify the call.

## Macros

- **Item macros must compose with attributes** like `#[derive(...)]` and `#[cfg(...)]`, and must handle `super::` resolving differently inside function bodies vs. module level.

## Drop

- **Drop implementations must never panic and should not block.** If cleanup might fail, provide an explicit `close()` method returning `Result`. The `Drop` impl should do best-effort non-blocking cleanup.

## Dependencies & Cargo

- **Feature names: no `use-`/`with-` prefixes.** Use `std` not `use-std`, `serde` not `with-serde`. Features must be purely additive — never use negative names like `no-std`.
- **Pin Git dependencies to a specific `rev`.** Without `rev`, `cargo update` can silently pull breaking changes.
- **Avoid default features** in Cargo.toml except for binary crates. Libraries should require explicit opt-in.
