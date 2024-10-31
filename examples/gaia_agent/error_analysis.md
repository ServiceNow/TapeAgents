# Gaia Validation Error Analysis

## TapeAgent v0.1 with GPT-4o-mini

### Overall results

- 165 tasks
- 11 error classes

| Level | Tasks | Anwer Produced | Correct Answer | Top 1 Error | Top 2 Error | Top 3 Error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 53 | 90% | 45% | bad reasoning | bad doc parsing | read page wrong  |
| 2 | 86 | 85% | 21% | bad reasoning |cannot find info | failed to follow plan |
| 3 | 26 | 69% | 0% | bad reasoning | failed to follow plan | read page wrong |
| *Total* | *165* | *84%* | *26%* | *bad reasoning* | *failed to follow plan* | *read page wrong* |


### Error classes
Bad reasoning - the model either produced incorrect chain of thoughts or chose the wrong action or action parameters. Sometimes the coding ability could help.

| Erorr Class | Number of Errors | Percentage |
| --- | --- | --- |
| bad reasoning | 34 | 20.6% | 
| failed to follow plan | 22 | 13.3% | 
| read page wrong | 19 | 11.5% | 
| cannot find info | 18 | 10.9% | 
| bad doc parsing | 14 | 8.5% | 
| bad plan | 8 | 4.9% | 
| cannot read video | 6 | 3.6% | 
| bad image reading | 4 | 2.4% | 
| wrong output format | 3 | 1.8% |
| browser errors | 2 | 1.2% |
| not enough steps | 2 | 1.2% |

Specific comments for tasks with bad reasoning:
- lack of modeling in code
- failed to follow instruction
- code could help
- code could help
- incorrectly rounded 41.48 to 42
- lack of work on subtask
- failed to check its own result
- failed to search for the older sale price
- found wrong info, did not accounted for the details of task
- hallucinated answer
- hallucinations
- hallucinations


## What we can do to improve

| Erorr Class | Fix Complexity | Idea |
| --- | --- | --- |
| bad reasoning | 3 | coding subagent could help partially |
| failed to follow plan | 2 | plan enforcing / multiagents could help |
| read page wrong | 2 | reflection could help partially |
| cannot find info | 3 | stuck in repeated search frequently, reflection could help |
| bad doc parsing | 1 | could be fixed by improving tools |
| bad plan | 3 | ? |
| cannot read video | 1 | could be fixed by improving tools |
| bad image reading | 1 | could be fixed by image qa subagent |
| wrong output format | 1 | could be fixed by special formatting node in the end |
| browser errors | ? | ? |
| not enough steps | ? | ? |


## Effects of the better model

Compare the results with the GPT-4o model and SFT SoTA submission:

| Level | Tasks | GPT-4o-mini | GPT-4o | SFT SoTA |
| --- | --- | --- | --- | --- |
| 1 | 53 | 24 (45%) | 25 (47%) | 31 (58.5%) |
| 2 | 86 | 18 (21%) | 30 (35%) | 40 (46.5%) |
| 3 | 26 | 0 (0%) | 1 (4%) | 7 (26,9%) |
| *Total* | *165* | *42 (26%)* | *56 (35%)* | *78 (47.3%)* |

Improvements by error class:

| Erorr Class | GPT-4o-mini errors | GPT-4o solved | Percent solved |
| --- | --- | --- | --- |
| bad reasoning | 34 | 12 | 35.3% |
| failed to follow plan | 22 | 3 | 13.6% |
| read page wrong | 19 | 2 | 10.5% |
| cannot find info | 18 | 3 | 16.7% |
| bad doc parsing | 14 | 3 | 21.4% |
| bad plan | 8 | 2 | 25.0% |
| cannot read video | 6 | 0 | 0.0% |
| bad image reading | 4 | 1 | 25.0% |
| wrong output format | 3 | 1 | 33.3% |
| browser errors | 2 | 1 | 50.0% |
| not enough steps | 2 | 1 | 50.0% |

Raw data available in [error_analysis.csv](scripts/error_analysis.csv)