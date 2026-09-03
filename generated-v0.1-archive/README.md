# Generated v0.1 archive

This directory preserves generated v0.1 source outside the Rust module tree.
It is archival evidence, not compiled v0.2 source.

| File | Bytes | Lines | SHA-256 |
|---|---:|---:|---|
| `hmm.rs.txt` | 18,790,144 | 569,177 | `169CEF54DD9EC0EDEC4C40B1FFCED69D67E602A51ECF536D7002D1B58F68F102` |
| `tslearn.rs.txt` | 1,887,283 | 60,208 | `1AD48FD80036752EC197BEF04C30C94A71C3BA74553B56908AA70411E973B25C` |
| `online.rs.txt` | 2,472,160 | 78,273 | `211C726A441D70A9B391F56EDDE3951BE5B5E10D8611A8B5A2CFC43826533EA2` |
| `coverage.rs.txt` | 681,720 | 23,295 | `9F2E4DDE55EAA12F617B3AA6DBC2D1B7CB67FBC109C1A55AC2CFCB3CF06FFD29` |
| `tsa.rs.txt` | 789,597 | 23,586 | `8A49C809C024B9DB41108EFDFA6AAD49CEF434B896CFF224798B1710E68786D7` |
| `stats.rs.txt` | 1,089,036 | 33,127 | `E1F72E0101D003E31117D008EC6CD723F2A1067AF0D45B06B042E191E1985ED8` |
| `iv.rs.txt` | 99,732 | 2,857 | `3238B87C68EED11FF199B7D4A7BDB4A79F140123252BEE87A40380A2A66E22DF` |
| `panel.rs.txt` | 150,783 | 4,205 | `CA2DB281CB4835CA59B3EE00352934620363FCE68A0B99FA935E96EB6AF5C74D` |

The files were moved byte-for-byte from `kanimiso/src/` on 2026-09-03.
The HMM generated families are replaced by the single runtime-parameterized
`hmm::HiddenMarkovModel<E>` implementation under `kanimiso/src/hmm/`.
The unverified tslearn surface had no workspace consumers and remains archived
until individual algorithms have an independent oracle and a non-generated API.
The online monolith is replaced by the small verified estimator modules under
`kanimiso/src/online/`; the archive retains the dropped experimental and
generated APIs for source-level migration searches only.
The generated coverage inventory is replaced by the small explicit
Verified/Experimental ledger in `kanimiso/src/coverage.rs`.
The TSA monolith is replaced by exact ARMA kernels and the independently
checked GARCH-family modules under `kanimiso/src/tsa/`; unverified
forecasting and compatibility names remain searchable only in this archive.
The stats monolith is replaced by the 580-line `kanimiso/src/stats/` module,
which retains only the independently checked `ProcessMleFit` and
`process_mle` surface. The unverified IV and panel modules depended on dropped
stats routines and had no other workspace consumers, so their source remains
searchable only in `iv.rs.txt` and `panel.rs.txt`.
