# Skill Issues: Data-Centric Optimization of Lakehouse Agents

## Overview

In ['Skill Issues: Data-Centric Optimization of Lakehouse Agents'](https://arxiv.org/abs/2606.01185), the text content of an agent skill is treated as a hyperparameter: as such, it can be optimized.

To do so, one needs a dataset and an optimizer. We create a synthetic dataset through a loop of i) an LLM generating a candidate, ii) another LLM auditing it and iii) a human reviewing the final result. Once the dataset is established, we use [GEPA](https://github.com/gepa-ai/gepa) as the skill optimizer. 

## Code Architecture

While the paper was originally developed for the [Bauplan lakehouse](https://www.bauplanlabs.com), we made the generation/optimization framework general and not tied to any Bauplan-specific construct. 

Hence, we divide the source code into:
1. The domain-agnostic generation/optimization engine, in `src/core`;
2. The domain-specific deatails, in `src/domains`. 

As a reference implementation, we provide the Bauplan case described in the paper in `src/domains/bauplan_lakehouse`. If you are interested in reproducing the exact Bauplan setup, [reach out](https://www.linkedin.com/in/jacopotagliabue/) for the prompt variables. 

## Quick Start

The project has local and cloud components. 

Locally, it requires Python 3.12 or newer and is managed with [uv](https://docs.astral.sh/uv/). For simplicity, we use [just](https://github.com/casey/just) as our main entrypoint, though it is possible to run all commands through `uv`. 

Bootstrap the environment with:
```sh
just install
```
This command will install all the dependencies, and also set up a `.env` file for you to fill in with LLM providers credentials. 

On the cloud, the project uses [`harbor`](https://github.com/harbor-framework/harbor) running on [Modal](https://modal.com) to optimize skills. For this, you will need a Modal account and its secrets stored in `.env`. 

Once the environment is set up, you can run `just` to inspect all the available commands. Use
```sh
just generate <domain> *args
```
to generate multiple dataset entries for a given domain. Then, run
```sh
just optimize <domain> <skill> *args
```
to optimize a skill for that domain based on the dataset you just created. Remember to use `--help` on each command for details on the available arguments.

## Adding Domains

A domain is added by creating a package under `src/domains/<name>/` that satisfies the conventions the _core_ relies on. The fastest way to start is

```sh
just new-domain <name>
```

which copies the shipped skeleton `src/domains/_template/` to `src/domains/<name>/` for you to fill in. Details on how to populate a new domain can be found [here](src/domains/_template/README.md).

## Citation

If you use this work, please cite our paper:

```bibtex
@misc{schneider2026skillissuesdatacentricoptimization,
      title={"Skill Issues'': Data-Centric Optimization of Lakehouse Agents}, 
      author={Nicole Rose Schneider and Davide Ghilardi and Giacomo Piccinini and Jacopo Tagliabue},
      year={2026},
      eprint={2606.01185},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.01185}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
