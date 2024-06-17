import re

def filter_potential_causes(response_text, predefined_causes):
    filtered_causes = []
    responded_potential_causes = re.findall(r'\d+\.\s*(.*)', response_text)
    for cause in predefined_causes:
        for responded_cause in responded_potential_causes:
            if cause.lower() in responded_cause.lower():
                filtered_causes.append(cause)
                break
    return filtered_causes
