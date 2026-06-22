# G_4kd08 Result Analysis

- Created: `2026-06-21 22:32:14 CST`
- Updated: `2026-06-22 12:35:19 CST`
- Status: `not promoted; validation pass did not transfer to test`

## Smoke01 Result

`G_4kd08_smoke01` restarted from A37 and used five validation gates to test whether simultaneous validation-side protection would transfer to the A37-aligned test gate.

Guard behavior:

- guard checks: `6`;
- accepted updates: `2`, steps `10, 12`;
- rolled back updates: `4`, steps `2, 4, 6, 8`;
- last validation check at step `12` passed all five gates.

A37-aligned best checkpoint:

- test PSNR `17.889656444940993`, delta `-0.000002008497823169364`;
- high-route `-0.11160577238467849`, delta `-0.000015728099650044736`;
- low-route `0.15914482675786737`, delta `+0.000008341377242176984`;
- A08-oracle `-0.19993999651505767`, delta `-0.00009900925367264946`;
- A15-oracle `0.2494341991175911`, delta `+0.00008062177568141093`.

A37-aligned final checkpoint:

- test PSNR `17.88963354114022`, delta `-0.00002491229859558075`;
- high-route delta `-0.000010635778976905708`;
- low-route delta `-0.00003568230462164257`;
- A08-oracle delta `-0.0000600524227276078`;
- A15-oracle delta `+0.0000050218812227553755`.

## Interpretation

G_4kd08 proved the validation guard can accept candidates that look simultaneous on the 24-batch validation view. The failure is transfer: the accepted candidates still miss A37 on test high-route and A08-oracle. Therefore the next probe needed positive margins above the validation baseline, not merely validation non-regression.

## Decision

`G_4kd08_smoke01` is a useful diagnostic but not promotable. Formal long training remains blocked.
