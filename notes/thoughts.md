# Overall Impression

- Compiler errors are very specific and helpful. The `rust-analyzer` plugin for VSCode (and other IDEs) is quite good at providing information about errors, pitfalls, and style mistakes while programming. I enjoyed this.
- A
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
  
  - The claim of an "algebriac type system" is too bold. Negative implementations and specialization are required to actually make it useful. 
- The Rust compiler is _very_ particular about specifying types and type bounds. It's difficult to get used to, but helpful.

```math
    x=2
```
