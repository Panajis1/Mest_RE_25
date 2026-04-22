
## 1. What Problem We Solve
In addition to knowing if a customer has solar panels (PV), it is increasingly important for grid operators to know if they also have a home battery system. Batteries change how a household interacts with the grid: they "hide" solar production and shift energy consumption to different times of the day.

This report describes a scientific method to:

Detect which PV-equipped customers likely have a battery.

Calculate the probability (confidence) of that detection.

Verify if the detection is physically possible given the size of their solar panels.

This allows for better grid stability forecasting, identifying "prosumers" who are self-sufficient, and targeting future flexibility services.


## 2. Data We Use
Smart meter data (Romande Energie) We look specifically at the Import (CONSO_KWH) during the evening hours (16:00 to midnight). This is when batteries typically discharge to power the home.

Weather data (MeteoSwiss) We use solar radiation and temperature. This is crucial because a battery can only charge if there was enough sun during the day.

PV Capacity Estimates We use the estimated size (kWp) of the customer's solar panels (from our previous PV report) to perform a "reality check."


## 3. How We Detect a Battery (The "Signature")
A household with a battery leaves a very specific "fingerprint" on the smart meter data. We find this by comparing sunny days to cloudy days that have the same outside temperature.

The Time Shift: On cloudy days, a house usually hits its peak electricity use around 18:00 or 19:00 (cooking, lights). On sunny days, a battery covers that peak. The "grid peak" then shifts to much later in the night (e.g., 21:00 or 22:00) only after the battery is empty.

The Energy Gap: On sunny days, there is a "gap" in evening imports. The house appears to be using almost no electricity from the grid for several hours, even though we know from cloudy days that the inhabitants are likely home and active.

## 4.  Physical Feasibility
A common error in data analysis is mistaking a "change in behavior" for a "battery." For example, if a family goes to the restaurant on a sunny Friday, their electricity use drops—looking like a battery discharge.

To solve this, our model uses the Laws of Physics:

The Solar Limit: We calculate exactly how much energy the customer's solar panels could have produced on that specific day based on the weather.

The Validation: If the "Energy Gap" in the evening is 10 kWh, but the solar panels could only have produced 4 kWh of surplus during the day, our model realizes a battery is physically impossible. It correctly flags this as a "behavioral shift" rather than a battery.

## 5. End to End Process
flowchart TD
    subgraph DataInputs [Data Inputs]
        Meter[15-min Smart Meter Data]
        Weather[Radiation & Temp]
        PV_Est[Estimated PV Size kWp]
    end

    subgraph Comparison [The Comparison]
        Match[Match Sunny vs Cloudy days by Temperature]
        Profile[Compare Evening Load Profiles]
    end

    subgraph Features [Detection Features]
        Shift[Peak Timing Shift: Is the peak delayed?]
        Gap[Energy Gap: Is evening import missing?]
    end

    subgraph Intelligence [Scientific Logic]
        PhysCheck[Physical Feasibility: Was there enough sun to fill a battery?]
        Logit[Logistic Regression: Calculate Probability 0-100%]
    end

    subgraph Results [Final Results]
        Flag[Battery: Yes/No]
        Confidence[Confidence Score %]
        Robustness[Validation Status: Robust/Doubtful]
    end

    Meter --> Match
    Weather --> Match
    Match --> Profile
    Profile --> Shift
    Profile --> Gap
    PV_Est --> PhysCheck
    Weather --> PhysCheck
    Shift --> Logit
    Gap --> Logit
    PhysCheck --> Logit
    Logit --> Flag
    Logit --> Confidence
    Logit --> Robustness

## 6. Outputs and What the numbers mean for decisions
Result,What it tells you,Business Application
**Battery Yes/No,Categorical classification of the customer.,Accurate reporting of storage penetration in the grid area.
Probability Score,"How ""certain"" the algorithm is (e.g., 95% vs 60%).","Selecting ""High-Certainty"" customers for pilot programs or technical audits."
Peak Shift (min),How many minutes the grid peak is delayed.,"Predicting ""Rebound Peaks""—when many batteries empty at once and hit the grid."
**Validation Status**,Whether the battery detection matches the solar size.,"Identifying customers with ""Doubtful"" status who might have unrecorded solar panels."


## 7. Limitations
The algorithm is highly robust for standard residential batteries. However, it may be less accurate for:

Very small batteries (e.g., 2 kWh) that empty before the evening peak is over.

Complex charging where the battery is charged from the grid (at night) rather than solar.

Significant behavior changes (e.g., extended vacations during sunny periods).