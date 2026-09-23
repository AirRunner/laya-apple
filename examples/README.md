# Examples

- `heterogeneous_routing.py` — submits a mix of short single-question, long, and
  multi-question requests concurrently to one `Laya` instance and prints which
  device each one ran on and why. Run with:

  ```bash
  uv run --extra ane python examples/heterogeneous_routing.py
  ```

  Works without the `[ane]` extra too; it just routes everything to the GPU and
  says so via `routing_reason`.
