#!/usr/bin/env bash
# Execute every numbered runner bare (no arguments) and sort the result into
# THREE outcomes, not two (task_wrong_output_repair, Part 3):
#
#   PASS        ran and exited 0.
#   NEEDS-ARGS  argparse rejected the bare invocation -- HEALTHY: the
#               runner is enforcing its required arms, not broken. Detected
#               from argparse's own signature: exit code 2 AND an
#               "error: the following arguments are required" line on
#               stderr, matched TOGETHER. Exit code 2 alone is not
#               specific enough -- it is Python's/argparse's generic
#               command-line-misuse code and can be produced by other
#               failures too, so it is necessary but not sufficient on its
#               own; the message text is what actually identifies "this
#               was argparse doing its job."
#   FAIL        anything else -- TypeError, NotImplementedError, an
#               unhandled traceback, or a non-zero exit that is not
#               argparse's own required-argument rejection.
#
# Before the nine-runner repair (docs/task_runner_repair.md) this script
# read "6 pass, 26 fail" -- but twelve of those "fail" runners were healthy
# runners correctly refusing to run without a stated configuration, scored
# as failures indistinguishably from a real defect. That made the health
# check unreadable: a genuine regression could hide among "expected"
# argparse rejections with no way to tell them apart from this summary line
# alone.
#
# WHAT THIS DOES NOT TEST. This is a WIRING smoke test only -- it catches
# import errors, missing constants, renamed attributes, and stale kwargs
# (the defect class that dominated the 2.6 correction), plus now censuses
# which runners enforce required arguments bare. It is NOT a correctness
# test: a PASS says the script ran to completion and exited 0, nothing
# about whether its output is right, and a NEEDS-ARGS says only that the
# runner's argparse block is wired up, not that the flags it demands are
# individually correct.
pass=0; needs_args=0; fail=0
for f in scripts/[0-9]*.py; do
  timeout 900 python "$f" > /tmp/smoke.log 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "  PASS        $f"; pass=$((pass+1))
  elif [ $code -eq 2 ] && grep -q "error: the following arguments are required" /tmp/smoke.log; then
    echo "  NEEDS-ARGS  $f"; needs_args=$((needs_args+1))
  else
    echo "  FAIL        $f  -- $(grep -m1 'Error\|error:' /tmp/smoke.log | cut -c1-80)"
    fail=$((fail+1))
  fi
done
echo; echo "  $pass passed, $needs_args needs-args, $fail failed"
