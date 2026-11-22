# Tower A Louver Prediction Workspace

This course project lets you compare daylight and solar energy predictions for one room and hour. It uses the repository’s **PVSD trial v0.2** model to evaluate four louver angles, **0°, 30°, 60°, and 90°**. Use the conversation to run predictions, understand results, compare scenarios, edit recommendation rules, and export selected angles.

## Run locally

You need the Python 3.11 model environment defined in [environment.yml](../environment.yml), Node.js 22.17 or later, and these two local data files.

- `ui/dataset/SEND_BIM_476Rooms.csv`
- `ui/dataset/SEND_ML_BIM_EPW_07-18.csv`

### First-time setup

Run the following from the repository root.

```bash
conda env create --prefix ./.conda/envs/pvsd-ml-v1 -f environment.yml
conda activate ./.conda/envs/pvsd-ml-v1
cd ui
python -m pip install -r requirements.txt
python -m server.catalog
npm ci
```

If you already have the project’s model environment, activate it and start with `cd ui`.

### Start a preview

From `ui/`, build the frontend and start the server. This serves the webpage and prediction API from the same process.

```bash
npm run build
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and click **Try an example** to run your first prediction.

### Develop with Vite

Run the Uvicorn command above in one terminal. In another terminal, start Vite from `ui/`.

```bash
npm run dev
```

Open the address printed by Vite. The default is [http://127.0.0.1:5173](http://127.0.0.1:5173).

### Configure the conversation agents

The backend reads `OPENROUTER_API_KEY` first. If the variable is unset, it can read a single OpenRouter key from `ui/key.md`. Git ignores `key.md` and local environment files. The credential stays on the backend.

To use an environment file, copy [`.env.example`](.env.example) to `.env`, fill in the key in your editor, and load the file in the terminal that will run the backend.

```bash
set -a
source ./.env
set +a
```

The agents share the model selected by `OPENROUTER_MODEL`, which defaults to `qwen/qwen3.5-flash-02-23`. Set `OPENROUTER_INTENT_MODEL`, `OPENROUTER_ROOM_MODEL`, or `OPENROUTER_EXPLAIN_MODEL` to choose a model for a specific role. Use a model that supports structured tool outputs and is available to your account and region. If OpenRouter accepts your key but the model returns HTTP 403 with a regional restriction, select a model available in your region.

Set `PVSD_LOCAL_ONLY=1` to run without calling OpenRouter. For example, use `PVSD_LOCAL_ONLY=1 python -m uvicorn server.app:app --host 127.0.0.1 --port 8000`. The prediction model runs locally. Common chat requests in English and Simplified Chinese use built-in parsing, and the same manual controls are available. This fallback also applies when a key is missing or a request times out. OpenRouter helps with a wider range of wording and generates explanations from prediction data.

### Interface and conversation language

The app opens in **English** by default. Use the top-right menu to switch to **Simplified Chinese**. The browser remembers your selection.

The conversation uses the interface language by default. Say **“Please reply in Chinese”** or **“Please reply in English”** to change the reply language for the current chat. You can include this instruction in the same message as a prediction request. A language-only request updates the reply setting so you can continue with the same prediction conditions.

English and Simplified Chinese chat prompts, clarification controls, and result cards work in local mode. A connected agent handles replies in other languages. Interface controls are available in English and Simplified Chinese.

## Use the workspace

- **Start with a room and time.** Try “BM223, June 15 at 2 pm, power first” or “Help me find a room on floor 5.” Each result card shows **Preference scores** on the left and **Predicted metrics** on the right. All five metrics are visible initially. The charts stack on narrow screens. Toggle individual lines in either legend, or hover over an angle to see its scores or original values. The tooltip follows the cursor within the browser edges and closes when the cursor leaves the plot. Click an angle to keep it highlighted in both charts. On a touchscreen, tap an angle to highlight it and show its tooltip. Select it again or press Escape to clear the selection. Choose a preference to emphasize its score line and mark the recommended angle. Teal and blue show daylight metrics, amber shows PV energy, and violet identifies the balanced score.
- **Check model details.** Hover over, focus, or tap **Model ready** to see the loaded model version, weather source, and available hours. A room without a simulation benchmark has an **Unsimulated room** badge by its conditions.
- **View all values.** Open this disclosure to see the five raw predictions, the absolute daylight score, each preference score, and the weights used. The predictions chart scales each metric to its own maximum across the four angles. Its tooltips and the table show the original values in lux, %, or Wh. The score chart compares angles within each preference.
- **Browse rooms.** The room picker lists all 476 rooms in the BIM catalog. Scroll through the room IDs or search by ID or floor. On a prediction card, the selected room appears first and the remaining rooms are ordered by ID. The **Simulated** label shows which rooms have a simulation reference.
- **Compare with simulation.** On a card with matching simulation data, select **Compare with simulation** in the predicted metrics chart. Use the metric buttons to show or hide pairs of lines. For each metric, the solid prediction line and dashed simulation line share one color and use the same peak across both sources and all four angles. The chart uses a single 0–100% axis. Hover over or tap an angle to see both original values and **Prediction − simulation** in lux, %, or Wh. Differences between percentages appear in percentage points. Select **Compare with simulation** again to return to the five-metric overview.
- **Change conditions or preference.** Select the room, date, or hour in a result card’s header to open one popover with all three fields. Change any combination of values, then select **Run prediction** to add one new card with the same preference. The original card keeps its conditions and results. **Cancel**, Escape, or clicking outside discards the edits. The first click on **Custom** opens the rule editor beside the button. After you apply a rule, **Custom** selects it again and the chevron opens it for editing. The rule editor uses a bottom sheet on narrow screens. Use **Enter room and time manually** to start without typing.
- **Name the chat.** Select the title in the top bar to rename it. Enter or the checkmark saves the name, and Escape or the close button cancels. The first prediction supplies a default name if you haven’t entered one.
- **Reopen a chat.** Open **Chats** next to **New chat** to see recent conversations, ordered by last edit. The checkmark identifies the current chat. This browser saves messages, prediction cards, bookmarks, and preferences as you work. **New chat** opens a blank conversation, and reloading the app reopens the last active chat.
- **Navigate the conversation.** Each bar along the left edge represents one prediction. Hover or focus for a preview with its conditions, preference, and recommended angles. Click the bar to jump to the card. On a touch screen, tap the bar and then the room name in the preview.
- **Bookmark and export predictions.** Bookmark a result from its card or preview. Click or tap the download icon beside the chat title. Choose from **Bookmarked** or **All**, then download the checked predictions from the current chat as CSV or JSON. The panel opens below the title on desktop and at the bottom of the screen on mobile. Batch exports include all four angles, with a prediction ID to distinguish each scenario.
- **Export selected angles.** Choose which angles to include and download a CSV or JSON file. The export includes all five predicted values, day of year and hour, model version and package hash, weather source, input coverage flags, scores, and recommendation markers. If you used a preference, it also includes its weights, lux targets, and cutoffs. Each row records whether the angle qualifies. The charts show energy in Wh, and exports use the model’s original unit of kWh.

## Supported natural-language operations

Ask for one main operation per message. Start with a room and time, then inspect, compare, or refine the results. You can ask for help or change the reply language at any point.


| Theme                         | Operation                          | Example request                                                                          | What happens                                                                                                                                                                      |
| ------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Run predictions**           | Find a room                        | “Help me find a room on floor 5.”                                                      | Shows matching rooms and a picker. You can provide the date and hour in the same message or later.                                                                                |
|                               | Run a prediction                   | “Predict BM223 for June 15 at 2 pm, balanced.”                                         | Predicts all four angles for the room and hour. The preference is optional.                                                                                                       |
|                               | Change conditions                  | “Try 3 pm instead.”                                                                    | Carries forward the confirmed room and date and adds a new result card. You can also change the room or date.                                                                     |
| **Inspect results**           | Show or hide chart metrics         | “Show only PV energy and useful daylight.” or “Restore all metrics.”                 | Changes which lines appear in the predicted metrics chart.                                                                                                                        |
|                               | Inspect an angle or open the table | “Highlight 60°.” or “Open the values table.”                                        | Highlights the angle across both charts or expands the values table. Say “Close the values table” to collapse it.                                                               |
|                               | Explain a result                   | “Why is this angle recommended?” or “Why is P10 lower at 60° than 0°?”             | Uses the referenced prediction. When you name a metric and angles, it focuses on their values and difference. **Explain** in the card header explains the current recommendation. |
| **Compare results**           | Compare angles                     | “Compare 0° and 60° on daylight.”                                                    | Compares the four daylight metrics. Name a metric such as P10 for one row, or ask without a metric to show all five.                                                              |
|                               | Compare scenarios                  | “Compare this with 3 pm for useful daylight.”                                          | Uses a matching result or predicts the requested scenario, then compares the selected metrics at the same angles.                                                                 |
| **Filter and rank angles**    | Set a limit and sort               | “Which angle generates the most PV with at least 40% coverage at 300 lux?”             | Filters the active prediction using the requested threshold, then sorts by PV energy. If none qualify, shows the closest angle and the shortfall.                                 |
| **Refine the recommendation** | Choose a preset preference         | “Use daylight first.”                                                                  | Updates the active recommendation or sets the preference for the next prediction if the chat has no result yet. The presets are Power first, Daylight first, and Balanced.        |
|                               | Customize scoring                  | “Use 70% useful daylight and 30% coverage at 300 lux.” | Opens the **Custom** editor on the relevant result card with the proposed weights. Preview the recommendation, then apply the rule. |
| **Export results**            | Export a result                    | “Export 30° and 60° as JSON.”                                                        | Opens the export dialog with the requested angles checked and the format highlighted. CSV and JSON are available.                                                                 |
| **Help and language**         | Ask about the project              | “What is P10?” or “Where does the weather data come from?”                           | Answers using project information about metrics, scoring, room coverage, weather, or supported dates and hours.                                                                   |
|                               | Change the reply language          | “Please reply in Chinese.”                                                             | Changes the conversation language for this chat. You can include the language request with another operation.                                                                     |

Follow-ups such as “this one” use the active result. Jump to a card from the prediction rail, use a card’s controls, or name a room, date, or hour to refer to a specific result. If multiple cards match, the chat asks you to choose. When comparing scenarios, use their raw values because each scenario’s preference scores are scaled to its own best angles.

Rank angles using the five raw metrics in a prediction. Give thresholds in lux, %, Wh, or kWh and join multiple limits with “and.” A limit such as “at least” includes equality, while “above” is strict. The ranking uses the requested metric. If no angle meets every limit, the app prefers the angle that misses the fewest limits, then the one with the smallest total relative shortfall.

## Ambiguity-aware interaction

The ambiguity-aware module checks the user’s request against the conversation state, room catalog, and supported ranges. It keeps confirmed information and sends unresolved fields or result references to the UI. The UI uses different controls for different ambiguities. A room search helps when the user lacks an ID, while buttons work for a short list of hours or cards.


| Situation                                                                            | What the app shows                                                                               |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| The room is missing or the user only knows the floor, such as “floor 5”            | Matching rooms with room ID, floor, and area, plus a search field                                |
| The room ID is unknown or the user gives several room IDs                            | A room selector. Any other confirmed fields carry over                                           |
| The date is missing or vague, such as “next weekend”                               | A month and day picker. The app converts the date to day of year using a 365-day simulation year |
| The time is vague or falls between whole hours, such as “afternoon” or “2:30 pm” | Buttons for the supported whole hours from 07:00 to 18:00                                        |
| The user gives two dates or times                                                    | Valid candidates to choose from                                                                  |
| The date or hour is out of range                                                     | The allowed range and a control for selecting a valid value                                      |
| Only the preference is missing                                                       | A prediction with preference buttons on the result card                                          |
| Several prediction cards match a room or time                                        | Buttons labeled with each candidate’s room, date, hour, and preference                          |
| An angle comparison lacks two distinct angles                                        | A picker for the four supported angles                                                           |
| Edited daylight weights do not add up to 100%                                        | An inline editor with the changes filled in and the current total visible                        |

The backend uses exact text matches for explicit room IDs, dates, and whole hours first. It calls the prediction service once the inputs are valid and the required data is available. The room picker uses floor and area from the building information model (BIM). Room orientation uses the model’s coordinate system. Filtering by geographic direction requires a verified coordinate mapping.

## Agent architecture

React and TypeScript render the conversation, cards, and controls. MobX stores the thread, active result, confirmed conditions, pending clarification, and card settings. FastAPI provides the room catalog, chat, prediction, and explanation endpoints. Pydantic AI connects the agents to OpenRouter.


| Role               | Responsibility                                                                                                                         |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Orchestrator       | Reads the latest message, confirmed conditions, and recent result references. Returns a typed action with any fields the user supplied |
| Room matching      | Narrows a shortlist of catalog rooms using available IDs, floors, and areas                                                            |
| Result explanation | Answers a question using the selected prediction, the current rule, and its calculated scores                                          |
| Translation        | Translates short responses into an explicitly requested language when needed                                                           |

The orchestrator routes prediction, explanation, scoring, comparison, ranking, chart, export, and help requests. It returns an action plan. Python and TypeScript execute that plan after validating its references and parameters. Pydantic validates agent responses against the schema. The prediction service checks room IDs and time ranges against the catalog and weather data. The frontend validates scoring weights and retains the validated rule on the card. Chart controls, comparisons, and downloads use stored model outputs. Each turn routes one primary action. The application keeps intermediate state when a request needs clarification.

The conversation replaces the separate scenario workspace. Editing conditions adds another prediction to the same thread, which makes the scenarios easy to find and compare. Editing preference applies a scoring rule to an existing result, which keeps the model output and user choices traceable.

## Recommended angles and scoring

The model predicts mean illuminance, P10 illuminance (the 10th percentile), the fraction of points at or above 300 lux, the useful daylight fraction (100–3,000 lux), and hourly AC solar energy. Both illuminance fractions use the room’s **spatial sample points** for that hour as the denominator.

The default daylight score uses the two fractions directly, with an editable starting weight of 60% for useful daylight and 40% for 300 lux coverage.

```text
Daylight score = 60 × useful daylight fraction
               + 40 × fraction of points at or above 300 lux
```

Both inputs are between 0 and 1. An angle with 80% useful daylight and 50% coverage scores 68/100. The same values produce the same daylight score across scenarios with the same rule.

The **Custom** editor provides a slider and an exact-value field for the daylight–PV mix and each daylight weight. Each daylight weight is independent. The running total shows how many percentage points you have left to assign or need to remove. For example, if a change brings the total to 105%, reduce any weight by five percentage points. The preview and **Apply rule** become available when the weights total 100% and the rule is valid.

Give a positive-weight P10 or mean illuminance term a lux target. Its contribution is its weight multiplied by `min(predicted lux / target lux, 1)`, so it earns full credit at the target.

The editor shows the current and proposed recommendations. **Preview details** shows the formula and each angle’s daylight score. **Apply rule** updates the charts and recommendation using the existing predictions. Dismissing the editor keeps the draft on that card. **Cancel** discards the draft. A scoring request in chat opens the same editor with the proposed values filled in.

Chat and the widgets edit the same draft rule. For example, say “Use 40% useful daylight, 30% coverage at 300 lux, 20% P10 with full credit at 100 lux, and 10% mean illuminance with full credit at 500 lux.” A complete set of daylight weights totaling 100% assigns zero weight to any omitted metric.


| Preference         | Weights                                                                                 |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **Power first**    | 100% solar energy                                                                       |
| **Daylight first** | 100% daylight                                                                           |
| **Balanced**       | 50% daylight and 50% solar energy                                                       |
| **Custom**         | User-defined daylight weights, lux targets, eligibility cutoffs, and daylight-to-PV mix |

Use chat when you need a hard limit, such as “Require P10 of at least 80 lux.” The editor shows each limit as a removable tag and shows which angles qualify in the preview. The preference score scales the daylight score and solar energy separately to their respective best values among qualifying angles, then applies the daylight and energy weights. A metric contributes zero if every qualifying angle has a zero value. Angles share the recommendation when they tie within the floating-point tolerance. When every angle falls short, the card shows the closest angle and its shortfall.

## Data preparation and model coverage

`python -m server.catalog` extracts **4,380 hourly records** from the 175,200-row combined source and writes `ui/dataset/weather_hourly.csv`. It uses `(DOY_1based, Hour)` as the key, checks that the five weather and solar-position values agree for all records at the same time, and validates day of year against the source hour index. It requires complete coverage from 07:00 to 18:00 each day of the 365-day simulation year.

At startup, the backend indexes the hourly weather table and the BIM catalog in memory. The interface labels the weather source **Original simulation weather**. No actual weather year was recorded in the data handoff. The backend indexes simulation results from the combined source when first needed and matches reference values by room ID, day of year, hour, and angle. These values come from the building simulation. The prediction result still comes from the local model.

All 476 room IDs in the BIM catalog passed input validation. The original simulation covers 10 rooms. Results for the other 466 rooms are labeled **Unsimulated room**, so they have no room-specific simulation benchmark. Seven rooms were used for model fitting, and three other simulated rooms contributed to model selection. Exports include coverage flags. See the [model usage guide](../docs/trial_v0_2_usage.md) and [methods](../METHODS.md).

To keep the data in a different local directory, run `python -m server.catalog <directory>` and point the backend’s `PVSD_DATASET` variable at the same directory.

## Verification

Run these commands from `ui/`.

```bash
python -m unittest server.test_workspace
npm test
npm run build
```

The checks cover bilingual behavior, local explanations, typed agent responses, unclear dates and times, result references, scoring edits, chart and export requests, and model compatibility with the released v0.2 examples.

For a course evaluation, use a fixed prompt set across the supported request types. Record whether the orchestrator routes each prompt correctly, whether the UI asks for the right missing information, and how many turns users need to complete a task. Record the actual model ID for any live-provider evaluation. Test the same core tasks with `PVSD_LOCAL_ONLY=1` to measure what the local fallback covers.
