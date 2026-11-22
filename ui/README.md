# Tower A Louver Prediction Workspace

This course project lets you compare daylight and solar energy predictions for one room and hour. It uses the repository’s **PVSD trial v0.2** model to evaluate four louver angles, **0°, 30°, 60°, and 90°**. You can add a recommendation preference, explore a scenario in the sidebar, or export selected results.

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

Set `PVSD_LOCAL_ONLY=1` to use local mode. Local parsing and manual controls also remain available when the key is missing or an agent request fails. [ff: what does this mean?]

### Interface and conversation language

The app opens in **English** by default. Use the top-right menu to switch to **Simplified Chinese**. The browser remembers your selection.

The conversation uses the interface language by default. Say **“Please reply in Chinese”** or **“Please reply in English”** to change the reply language for the current chat. You can include this instruction in the same message as a prediction request. A language-only request updates the reply setting so you can continue with the same prediction conditions.

English and Simplified Chinese chat prompts, clarification controls, and result cards work in local mode. A connected agent handles replies in other languages. Interface controls are available in English and Simplified Chinese.

## Use the workspace

- **Start with a room and time.** Try “BM223, June 15 at 2 pm, power first” or “Help me find a room on floor 5.” Each result card shows **Preference scores** on the left and **Predicted metrics** on the right. All five metrics are visible initially. The charts stack on narrow screens. Toggle individual lines in either legend, or hover over an angle to see its scores or original values. The tooltip follows the cursor and can cross into the other chart. Its position is clamped at the browser edges. Click or tap to pin that angle across both charts. Select it again or press Escape to clear it. Choose a preference to emphasize its score line and mark the recommended angle. Teal and blue show daylight metrics, amber shows PV energy, and violet identifies the balanced score.
- **View all values.** Open this disclosure to see the five raw predictions, the absolute daylight score, each preference score, and the weights used. The predictions chart scales each metric to its own maximum across the four angles. Its tooltips and the table show the original values in lux, %, or Wh. The score chart compares angles within each preference.
- **Explore a scenario.** Open the **Explore** sidebar or start from a result card. Click **Recalculate** after changing the room, date, or hour. Preferences and scoring weights update the ranking immediately. The sidebar keeps its own scenario. Click **Add to chat** to create a new result card and continue the conversation from that scenario.
- **Export selected angles.** Choose which angles to include and download a CSV or JSON file. The export includes all five predicted values, day of year and hour, model version and package hash, weather source, input coverage flags, scores, and recommendation markers. If you used a preference, it also includes the weights and P10 target. The charts show energy in Wh, and exports use the model’s original unit of kWh.

## How the app handles unclear requests

The app extracts the conditions stated in the latest message. It checks them against conditions already confirmed in the conversation, the room catalog, and the valid input ranges. Each field can be confirmed, missing, ambiguous, one of several candidates, or out of range. The backend returns the field status and candidate values, and the interface displays the appropriate control. Selecting a value updates that field directly.


| Situation                                                                            | What the app shows                                                                               |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| The room is missing or the user only knows the floor, such as “floor 5”            | Matching rooms with room ID, floor, and area, plus a search field                                |
| The room ID is unknown or the user gives several room IDs                            | A room selector. Any other confirmed fields carry over                                           |
| The date is missing or vague, such as “next weekend”                               | A month and day picker. The app converts the date to day of year using a 365-day simulation year |
| The time is vague or falls between whole hours, such as “afternoon” or “2:30 pm” | Buttons for the supported whole hours from 07:00 to 18:00                                        |
| The user gives two dates or times                                                    | Valid candidates to choose from                                                                  |
| The date or hour is out of range                                                     | The allowed range and a control for selecting a valid value                                      |
| Only the preference is missing                                                       | A prediction with preference buttons on the result card                                          |

The backend uses exact text matches for explicit room IDs, dates, and whole hours first. It calls the prediction service once the inputs are valid and the required data is available. The room picker uses floor and area from the building information model (BIM). Room orientation uses the model’s coordinate system. Filtering by geographic direction requires a verified coordinate mapping.

## Agent architecture

React and TypeScript build the interface. MobX stores the conversation, current conditions, predictions, and sidebar state. FastAPI provides the room catalog, chat, prediction, and explanation endpoints. Pydantic AI connects the agents to OpenRouter. The Python backend uses the state of the input fields to decide which agent to call.

[ff: I think we may need to have an orchestrator agent to support freeform natural language request, as not every request is about prediction, user may request for explaining a prediction result, asking to customize the preference calculation, currently, the chat interface is too limited to just predict. what kind of requests do you think we should cover? I am considering removing the side panel, as it's kind of conflicting with the conversational based ui design.]


| Agent              | Input and output                                                                                                            | When it runs                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Intent             | Reads the latest message and confirmed conditions. Returns structured candidates for time, preference, and room description | When the user sends a natural-language message                        |
| Room matching      | Reads a room description and a short list of catalog matches. Returns eligible room IDs                                     | When the description is ambiguous and the catalog has candidates      |
| Result explanation | Reads prediction values and computed scores. Returns a short explanation                                                    | When the user clicks **Explain results** or asks a follow-up question |

Pydantic validates each agent’s output structure. Application rules then validate the room ID, date, and hour. Agent calls have time and retry limits. Room and weather lookups, model loading, regression, and ranking all run in deterministic code. The frontend recalculates scores immediately when sidebar weights change. For an explanation request, the backend applies the same scoring rules before passing results to the agent.

## Recommended angles and scoring

The model predicts mean illuminance, P10 illuminance (the 10th percentile), the fraction of points at or above 300 lux, the useful daylight fraction (100–3,000 lux), and hourly AC solar energy. Both illuminance fractions use the room’s **spatial sample points** for that hour as the denominator.

The default daylight score is calculated as follows.

```text
Daylight score = 50 × useful daylight fraction
               + 30 × fraction of points at or above 300 lux
               + 20 × min(P10 / target illuminance, 1)
```

The default P10 target is 100 lux. For the preference score, the app divides the daylight score and solar energy by their respective maximum values across the four angles. It then applies the current daylight and energy weights to give a score from 0 to 100. If a metric is zero for all four angles, its contribution is zero.


| Preference         | Weights                                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Power first**    | 100% solar energy                                                                                                                                    |
| **Daylight first** | 100% daylight                                                                                                                                        |
| **Balanced**       | 50% daylight and 50% solar energy by default                                                                                                         |
| **Custom**         | Adjust the daylight and energy weights, the three daylight metric weights, and the P10 target. The three daylight metric weights must add up to 100% |

Angles share the recommendation when their highest scores agree within the floating-point tolerance. Use the preference score to compare angles within the same room and hour. The interface also shows the absolute daylight score and solar energy values.

The weights and target are interface design settings. A course report can explain how they represent user priorities and how to reproduce a recommendation from the same prediction values.

## Data preparation and model coverage

`python -m server.catalog` extracts **4,380 hourly records** from the 175,200-row combined source and writes `ui/dataset/weather_hourly.csv`. It uses `(DOY_1based, Hour)` as the key, checks that the five weather and solar-position values agree for all records at the same time, and validates day of year against the source hour index. It requires complete coverage from 07:00 to 18:00 each day of the 365-day simulation year.

At startup, the backend indexes the hourly weather table and the BIM catalog in memory. The interface labels the weather source **Original simulation weather**. No actual weather year was recorded in the data handoff.

All 476 room IDs in the BIM catalog passed input validation. The original simulation covers 10 rooms. Results for the other 466 rooms are labeled **Unsimulated room**, so they have no room-specific simulation benchmark. Seven rooms were used for model fitting, and three other simulated rooms contributed to model selection. Exports include coverage flags. See the [model usage guide](../docs/trial_v0_2_usage.md) and [methods](../METHODS.md).

To keep the data in a different local directory, run `python -m server.catalog <directory>` and point the backend’s `PVSD_DATASET` variable at the same directory.

## Verification

Run these commands from `ui/`.

```bash
python -m unittest server.test_workspace
npm test
npm run build
```

The checks cover bilingual behavior, local explanations, offline responses from the three agent roles, optional preferences, unclear dates and times, out-of-range inputs, conditions carried across messages, weather lookup coverage, and ranking in Python and TypeScript. They also compare 28 predictions against the released v0.2 examples. Floating-point differences on macOS were within the test tolerance.

Chrome checks used real prediction results to verify clarification controls, results without a selected preference, chart highlighting, weight changes, adding a sidebar scenario to the chat, and exporting selected rows to CSV and JSON. They also checked the mobile sidebar, the English default, saved language preferences, and Chinese replies and controls inside the English interface.

Live checks of the default OpenRouter model covered English intent parsing, room matching, explanations in English and Chinese, and French confirmations, prediction introductions, and explanations. For course report evaluation, use a fixed set of Chinese prompts to measure first-turn parsing accuracy, control selection accuracy, and the number of clarification turns. Test manual mode with rate limits or a simulated timeout. Record the actual model ID with any live agent results.
