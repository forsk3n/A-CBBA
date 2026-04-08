\# A-CBBA



\## About

This project contains modifications to the algorithm from the paper \*\*"A Two-Level Clustered Consensus-Based Bundle Algorithm for Dynamic Heterogeneous Multi-UAV Multi-Task Allocation"\*\*. The original source code and research were published by the authors.



\## Link to the original article

> Chao, Y.; et al. A Two-Level Clustered Consensus-Based Bundle Algorithm for Dynamic Heterogeneous Multi-UAV Multi-Task Allocation. \*Sensors\* \*\*2025\*\*, \*25\*, 6738. \[[https://doi.org/10.3390/s25216738](https://doi.org/10.3390/s25216738)](https://doi.org/10.3390/s25216738)



\## Link to the original code

The source code of the original TLC-CBBA algorithm is available in the following repository:

\[[https://github.com/ycchao0406/TLC\_CBBA](https://github.com/ycchao0406/TLC\_CBBA)](https://github.com/ycchao0406/TLC_CBBA)



\## Modifications made



In this version of the algorithm, the following changes have been introduced:



\*   `acbba_core.py` – Implements \*\*A-CBBA\*\*, an extension of the original CBBA with:

&#x20;   - \*Adaptive consensus stopping\*: each agent monitors local bid‑table stability over a sliding window of `τ` rounds; the algorithm terminates as soon as all reachable agents converge, avoiding both premature stops (under packet loss) and wasteful rounds (under ideal channels).

&#x20;   - \*Weighted relay selection\*: when a direct link is lost, messages can be forwarded through a relay agent chosen by a score that combines the relay’s residual fuel (reliability proxy) and the quality of the channel toward the destination.

&#x20;   - \*Per‑round packet‑loss simulation\*: each edge of the communication graph is independently dropped with probability `p\_loss` before building the effective routing topology.



\*   `run_experiment.py` – Provides a \*\*validation framework\*\* that compares baseline CBBA against A‑CBBA under increasing packet loss (0% to 50%). It runs Monte‑Carlo simulations, computes three metrics (task completion rate, consensus rounds, conflict‑free rate), and generates:

&#x20;   - `results/tcr_vs_ploss.png`

&#x20;   - `results/rounds_vs_ploss.png`

&#x20;   - `results/conflict_vs_ploss.png`

&#x20;   - `results/summary_table.csv`



\## Acknowledgements

I thank the authors of the original algorithm for their work and for making the source code publicly available.



\## License

The original source code is publicly available as stated in the Data Availability Statement of the article:  

\*"All data supporting the findings of this study are contained within the article... the corresponding source code has been made publicly available at [https://github.com/ycchao0406/TLC_CBBA](https://github.com/ycchao0406/TLC_CBBA) "\*  



No explicit open-source license (e.g., MIT, GPL) is provided for the original repository. This project is a fork of that code. Unless a license is added by the original authors, all rights remain with them. If you intend to use or redistribute this modified version, please consult the original authors for clarification.

