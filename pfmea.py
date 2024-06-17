# risk_priority_number = severity * occurance * detectability

def get_severity(cause):
    severity_scores = {
        "fungi": 7,
        "over watering": 5,
        "high humidity": 3,
        "poor air circulation": 2,
        "insufficient sunlight": 4,
        "nutrient deficiency": 6,
        "insect infestation": 5,
        "soil pH imbalance": 4,
        "contaminated tools": 5,
        "poor soil drainage": 5
    }
    return severity_scores.get(cause, 5)

def get_occurrence(cause):
    occurrence_scores = {
        "fungi": 7,
        "over watering": 5,
        "high humidity": 3,
        "poor air circulation": 2,
        "insufficient sunlight": 3,
        "nutrient deficiency": 4,
        "insect infestation": 6,
        "soil pH imbalance": 4,
        "contaminated tools": 3,
        "poor soil drainage": 5
    }
    return occurrence_scores.get(cause, 5)

def get_detectability(cause):
    detectability_scores = {
        "fungi": 7,
        "over watering": 5,
        "high humidity": 3,
        "poor air circulation": 2,
        "insufficient sunlight": 3,
        "nutrient deficiency": 4,
        "insect infestation": 6,
        "soil pH imbalance": 3,
        "contaminated tools": 5,
        "poor soil drainage": 3
    }
    return detectability_scores.get(cause, 5)
  

def perform_pfmea(potential_causes):
    pfmea_result = []
    for cause in potential_causes:
        severity = get_severity(cause)
        occurance = get_occurrence(cause)
        detectability = get_detectability(cause)
        rpn = severity*occurance*detectability    # rpn = risk priority number
        pfmea_result.append((cause, rpn))
    pfmea_result.sort(key=lambda x: x[1])
    root_cause = pfmea_result[0][0]


    return root_cause

# potential_causes = ["over watering", "fungi", "high humidity", "insufficient sunlight"]
# root_cause = perform_pfmea(potential_causes)
# print(root_cause)
