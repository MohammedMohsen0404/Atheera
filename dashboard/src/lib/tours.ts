/** The guided tour for each page. A step highlights the part marked data-tour="…" and explains it. */
export interface TourStep {
  /** Which part to highlight (its data-tour name). Leave it out for a step that is only a message. */
  target?: string;
  title: string;
  text: string;
}

const at = (name: string) => `[data-tour="${name}"]`;

export const TOURS: Record<string, TourStep[]> = {
  "/": [
    {
      title: "Welcome to Atheera",
      text: "Atheera reads breathing and movement from the Wi-Fi already on board. No cameras and no wearables. This short tour shows what each part of this page does. Use the arrow keys or the buttons, and press Esc to leave at any time.",
    },
    {
      target: at("replay"),
      title: "The replay controls",
      text: "Everything here is a replay of recorded Wi-Fi data, played as if it were live. The ten-minute shift starts by itself. Pause or play with the button (or Space), change the speed, or restart it. T+ is the time since the start.",
    },
    {
      target: at("menu"),
      title: "Four pages",
      text: "Crew is this page. Station is a 3D view of the station. Shift log lists every alert. How it works explains the method and the results. The red number is how many alerts are still open.",
    },
    {
      target: at("crew-list"),
      title: "The crew",
      text: "One tile per person, with a live wave read from the Wi-Fi and their latest reading. Click a tile, or press 1 to 7, to choose who you are looking at. A tile changes colour when that person needs attention.",
    },
    {
      target: at("figure"),
      title: "The selected person",
      text: "The lungs move with the breathing measured from the Wi-Fi signal. The labels show what is measured: breathing, posture and movement. The badge in the corner is their status.",
    },
    {
      target: at("wifi"),
      title: "The Wi-Fi signal",
      text: "The raw signal over the last 12 seconds, one row per frequency. Breathing and movement make ripples in it, and everything else on this page is read from those ripples.",
    },
    {
      target: at("activity"),
      title: "Activity",
      text: "What the person is doing, and how sure the model is about each activity. In rooms that only measure breathing it shows what tonight's schedule says, such as asleep.",
    },
    {
      target: at("rest"),
      title: "Rest in the sleep period",
      text: "Under Activity, for crew in their sleep period: how much of it they spent resting, meaning steady breathing and no movement for 30 seconds. It is rest, not sleep stages. Time the Wi-Fi could not judge is shown separately and never counted as rest. Movement nearby, while the person's own breathing stays steady, does not count against them.",
    },
    {
      target: at("breathing"),
      title: "Breathing",
      text: "The breathing rate in breaths per minute, the normal range for what the person is doing, and the chest-belt reading used as a check when there is one. If the signal is unclear it says so instead of guessing.",
    },
    {
      target: at("scenarios"),
      title: "Try a scenario",
      text: "Play out an event for the selected person and watch the system react. A simulated fall raises an emergency within seconds. Radio interference makes the signal noisier; breathing is still read, so it raises at most an advisory, never an alarm.",
    },
    {
      target: at("timeline"),
      title: "Tonight",
      text: "The whole shift, one line per person. Diamonds are scheduled events and ticks are alerts. Click anywhere on it to jump to that moment.",
    },
    {
      title: "When something goes wrong",
      text: "A new warning or emergency appears beside the heading for a few seconds. It then stays open in the Shift log, and in the red number in the menu, until someone presses Acknowledge.",
    },
    {
      title: "What you are watching",
      text: "Everything in this demo is recorded Wi-Fi data from real people, replayed as if it were live and run through the same pipeline a live stream would use. The recordings were made on Earth, not in orbit, and breathing and activity come from two public datasets. Atheera is a research prototype, not a medical device. Press the question-mark button to take this tour again.",
    },
  ],

  "/station": [
    {
      target: at("station-view"),
      title: "The station",
      text: "A 3D schematic of the crew quarters, the lab and the galley. You can see inside the modules, and each person acts out what the data says: asleep in a sleeping bag, walking, running on the treadmill, or falling. Drag to orbit, and hold Ctrl while scrolling to zoom.",
    },
    {
      target: at("station-tools"),
      title: "The camera",
      text: "Whole station resets the view and Auto orbit circles slowly. Walk through puts you inside the station: move with W A S D, look around by dragging, and press Esc to come back out.",
    },
    {
      target: at("station-panel"),
      title: "Details for one person",
      text: "Pick a person to see how Atheera reads them step by step: the raw Wi-Fi signal, the breathing wave, then the rate and how clear it is.",
    },
    {
      title: "Warnings in 3D",
      text: "When someone has a warning or an emergency, their name tag is outlined and the lights in their module flash.",
    },
  ],

  "/log": [
    {
      target: at("log-list"),
      title: "The alerts",
      text: "Every alert says who, where, what happened, the evidence behind it and what to do next. Press Acknowledge once you have seen it.",
    },
    {
      target: at("log-tabs"),
      title: "Filter",
      text: "Switch between alerts that are still open, ones already handled, and everything.",
    },
    {
      target: at("levels"),
      title: "Alert levels",
      text: "From a degraded signal up to an emergency, with what each level means and what to do about it.",
    },
  ],

  "/how": [
    {
      target: at("how-steps"),
      title: "Five steps",
      text: "From the Wi-Fi channel to an alert for the flight surgeon: listen, clean the signal, read the body, decide, and tell the surgeon.",
    },
    {
      target: at("how-results"),
      title: "How well it works",
      text: "The headline results. Each one shows the range it could really be, because small samples carry real uncertainty.",
    },
    {
      target: at("how-models"),
      title: "What is inside",
      text: "Each part, the method it uses, and whether it was learned from data. The breathing rate is deliberately not learned.",
    },
    {
      target: at("how-notes"),
      title: "Read these before quoting a number",
      text: "What Atheera does not claim, the limits of the evidence, how the results were checked, and where the data comes from.",
    },
  ],
};
