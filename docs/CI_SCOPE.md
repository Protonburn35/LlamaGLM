# CI scope and follow-up

Status: workflow cleanup is deferred. The inherited upstream workflows still have their original automatic triggers.

Pushing the v0.8.0 import to `master` launched dozens of llama.cpp workflows, including Android, Apple, self-hosted runners, other accelerator backends, and release builds. That fan-out was excessive for this Windows application. Cancelling those runs did not remove the triggers, so another ordinary push can repeat it.

Before the next source push, inspect the workflow triggers and avoid launching the entire upstream matrix accidentally. For a focused fix, `[skip ci]` on the commit can suppress push-triggered workflows; explicitly dispatch only the relevant workflows afterward. Do not describe skipped or cancelled checks as passed. A rerun uses the original run's source commit, not the latest branch tip.

The proposed future configuration is:

- Routine checks: native Windows CPU, CUDA SM89/SM120 compilation, CPU/RPC/launcher regression tests, and basic code/style checks.
- Retain AMD GPU and Vulkan coverage. These remain useful compatibility targets; consider manual runs or a separate limited schedule instead of removing their support.
- Other inherited platforms, self-hosted matrices, and publication pipelines should run only when relevant and intentionally requested.

Do not implement this cleanup without checking the current request: it was explicitly postponed during the import review. Preserve backend source code and tests when changing CI triggers. The inherited `build-cann.yml` has no executable jobs, which produces a workflow-validation failure; that is separate from an application build failure and needs attention during cleanup.

Always distinguish successful Git pushes, local CPU tests, hosted Windows compilation, and actual GPU/full-model execution in status reports. The original VM build does not validate later source changes.
