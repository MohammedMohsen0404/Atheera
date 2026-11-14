# Atheera and the Space Apps 2026 challenge

Challenge: **Create Health Monitoring Software for Astronauts on Space Missions** (NASA Space Apps 2026,
[challenge page](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/)).

The challenge asks for a visual tool that gathers health indicators so astronauts can evaluate their health and act on it.
Atheera answers it with the indicators that need **nothing worn and nothing to remember**, read from the Wi-Fi already
crossing a module and shown to the crew and to flight surgeons on the ground. It is built to run beside the medical kit
crews already use, not to replace it.

## What the challenge asks, and where Atheera answers it

| The challenge asks for | Atheera |
|---|---|
| **A clear visual tool** (a human figure, animations, interactivity) | A 3D crew figure and a 3D station view with a walk-through mode, live breathing charts, colour-coded status for every crew member and an alert timeline |
| **Daily sleep hours** | **Rest during the sleep period**, from Wi-Fi alone: time with steady breathing and no body movement, movements per hour, and time the system could not judge, shown separately and never counted as rest. This is rest time, not sleep stages. Breathing rate is added as a further indicator |
| **When a result warrants attention from the crew or flight surgeons** | Caution and warning levels (advisory, caution, warning, emergency) that follow the ISS alert classes. Every alert lists the evidence and confidence behind it and says what to do next, for example contact the crew member on the comm loop. Treatment decisions stay with the crew and flight surgeons |
| **How spaceflight hazards affect the indicators** | Each signal is tied to a NASA hazard (table below). Of NASA's five hazards Atheera addresses three: isolation and confinement, the closed environment and gravity fields. Radiation and distance from Earth are left to other tools |
| **Simple explanations for non-experts** | A guided tour on every page and a "How it works" page explain each signal, its results and its limits in plain words |
| **Color coding, dynamic charts over time** | Status colours per crew member, a live breathing waveform and a shift timeline |
| **Different environments** | The demo is a space-station night shift. The same sensing applies to any closed habitat with Wi-Fi, such as a transit vehicle or a surface base |
| Heart rate, blood pressure, blood sugar, body temperature, mood | Outside Atheera's scope by design. Wi-Fi sensing cannot measure these reliably, so they stay with dedicated instruments in the medical kit, which Atheera complements |

## NASA hazards and risks behind each signal

The hazard and risk names and the quotes come from the NASA and Canadian Space Agency reading pages listed in the challenge resources.
The challenge offers no datasets, so every measured number comes from public datasets (WiRe-Breath and UT-HAR), not from NASA.

| Atheera signal | NASA hazard | NASA risk and why it matters |
|---|---|---|
| Rest during the sleep period | Isolation and confinement; hostile/closed environment | [Risk from Inadequate Sleep and Irregular Schedules](https://www.nasa.gov/reference/risk-from-inadequate-sleep-and-irregular-schedules/): "Astronauts witness 16 sunrises and sunsets per day aboard the space station, which can disrupt their sleep-wake cycle." The [risks overview](https://www.nasa.gov/hrp/risks/) lists "chronic sleep loss" among the stressors |
| Breathing rate at rest | Hostile/closed environment | The [risks overview](https://www.nasa.gov/hrp/risks/) lists mild hypoxia, "when oxygen levels in body tissues become lower than normal". Atheera reports breathing rate, not oxygen |
| Activity, fall and abnormal-motion alerts | Gravity fields | [Risk of Injury from Dynamic Loads](https://www.nasa.gov/reference/risk-of-injury-from-dynamic-loads/): weakened bones and muscles "heighten injury risk" when crews move between gravity fields |
| Rest and activity patterns over a mission | Isolation and confinement | [Risk of Behavioral Conditions](https://www.nasa.gov/reference/risk-of-behavioral-conditions-and-psychiatric-disorders/): "These factors may impact a crew member's sleep, morale, and decision making" |

**Closest NASA study.** NASA's current ARCHeR study "monitored astronauts' well-being, activity, and sleep
patterns" on Artemis II with a worn actigraphy device ([ongoing HRP research](https://www.nasa.gov/hrp/ongoing-hrp-research/)).
Atheera's rest tracker uses the same scoring idea as actigraphy (low movement around each moment means rest),
but takes the movement from Wi-Fi, so nothing has to be worn, charged or remembered.

## How the rest measure was checked

`scripts/eval_rest.py` (results in `results/eval_rest.json`) plays every WiRe-Breath recording as a sleep
period. The people in those recordings lay or sat still, so any time not scored as rest is an error.

| | Quiet room | Person moving nearby | All |
|---|---|---|---|
| Scored as rest | 99.3% | 96.4% | **97.9%** |
| Scored as awake (false) | 0.0% | 0.0% | **0.0%** |
| Could not judge (shown separately) | 0.7% | 3.6% | 2.1% |

Two fixes made this possible:
- The receiver's automatic gain steps used to read as body movement: 17.3 false movements per hour. Dividing
  out each packet's overall gain cut that to 2.8.
- Movement during which the crew member's own breathing stays steadily tracked is attributed to someone nearby.
  It shows as an advisory, not a warning, and does not interrupt rest. False movements fell to 0.0 per hour.

The opposite check, an awake person lying still scored as resting, needs a dataset that pairs breathing with labelled
wake periods; none is public yet, which is why it is a planned step. See [LIMITATIONS.md](LIMITATIONS.md).

## Environments

Atheera's demo is a space-station night shift replayed from real recordings made on Earth. No CSI data from orbit
exists yet, so recordings in a metal-walled mock-up are an early step of the plan.
