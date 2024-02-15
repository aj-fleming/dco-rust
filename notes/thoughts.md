# Overall Impression

Investing more time into Rust as a language for scientific computing and AD doesn't offer significant enough improvements over C(++) or other languages that are a bit friendlier to the user.

The maturity of `dco/c++` eliminates any advantage Rust may have over C++. Developing any serious application in Rust that requires AD would require untold effort to match the flexibility and power of `dco/c++`. 

In terms of prototyping speed, Rust is as a slow as C++ to write. `Cargo`, however, makes it easy to test and deploy packages. A working proof-of-concept or similar project can be easily shared with others without losing a few hours to install libraries. I have, however, never used `conan` or `vcpkg`.

## The Good

- Compiler errors are very specific and helpful. The `rust-analyzer` plugin for VSCode (and other IDEs) is quite good at providing information about errors, pitfalls, and style mistakes while programming. I enjoyed this.
- The Rust compiler is _very_ particular about specifying types and type bounds. It's difficult to get used to, but helpful.
- Lifetime enforcement is nice when implementing adjoint... memory leaks can be easily avoided.

## The Bad

- Generic programming feels limited. There is no way to specialize interface implementations, which often requires more trait parameters than strictly necessary, especially compared to C++ templates or Julia's `::Any` type.
  - In order to avoid overlapping implementations of some simple routines (`passive_value`, `highest_order_derivative`), I found it simpler to write

  ```rust
  pub struct ForwardDiffDual<T ,U>{...}
  pub type Tangent<T> = ForwardDiffDual<T, T>
  ```

  rather than

  ```rust
  pub struct Tangent<T>{...}
  ```

- The claim of that rust has an "algebriac type system" is too bold. Negative implementations and specialization are required to actually make it useful. There are "experimental features" available that may be worth investigating.

## The Ugly

- Many components that should be part of the standard library (numerical type traits, approximate comparisons, etc...) are in external crates that require downloading.
- `.unwrap()`
- Type safety on methods leads to the writing of many annoying casts.
