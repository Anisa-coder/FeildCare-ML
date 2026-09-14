"""Agricultural-extension treatment guidance keyed to model class keys."""

DISCLAIMER = (
    "General guidance based on agricultural extension sources. Always follow product "
    "label instructions and consult a local agricultural extension officer before "
    "applying any chemical treatment."
)

SOURCES = [
    "NC State Extension",
    "University of Minnesota Extension",
    "University of Maryland Extension",
    "UC Integrated Pest Management",
    "UF/IFAS Extension",
    "Iowa State University Extension",
    "University of Arkansas Extension",
]


def treatment(name, classification, agent, causes, cure, prevention, label="Treatment Steps", note=None):
    return {
        "display_name": name,
        "classification": classification,
        "agent": agent,
        "cause_factors": causes,
        "cure_points": cure,
        "prevention_points": prevention,
        "management_label": label,
        "note": note,
        "disclaimer": DISCLAIMER,
        "sources": SOURCES,
    }


TREATMENTS = {
    "corn__cercospora_leaf_spot_gray_leaf_spot": treatment(
        "Corn Cercospora Leaf Spot / Gray Leaf Spot", "Fungal disease", "Cercospora zeae-maydis",
        ["High humidity above 95% for 24 hours or longer", "Warm temperatures and prolonged leaf wetness", "Continuous corn rotation", "Minimum-till fields with corn residue on the soil surface"],
        ["Apply a strobilurin- or triazole-class foliar fungicide between tasseling and early grain fill (VT–R2) if disease is severe.", "Till under crop residue after harvest to speed decomposition.", "Improve field drainage and airflow where possible.", "Scout weekly from silking onward for lesion spread.", "Remove severely infected lower leaves in high-value plantings."],
        ["Plant resistant or tolerant hybrids and check seed-company resistance ratings.", "Rotate with a non-host crop such as soybean or small grains for at least one season.", "Use conventional tillage in fields with a history of disease.", "Avoid continuous corn-on-corn planting.", "Maintain balanced fertilization and avoid excess nitrogen."],
    ),
    "corn__common_rust": treatment(
        "Corn Common Rust", "Fungal disease", "Puccinia sorghi",
        ["Cool-to-moderate temperatures", "Wind-dispersed spores", "Greater susceptibility in sweet corn and seed corn"],
        ["Apply a foliar fungicide if pustules appear early on susceptible sweet or seed corn.", "Remove and destroy heavily infected residue after harvest.", "Monitor closely while weather remains cool and humid."],
        ["Plant resistant hybrids; most modern field-corn hybrids carry resistance.", "Avoid unusually early or late planting in high-risk regions.", "Manage crop residue to reduce overwintering inoculum."],
    ),
    "corn__northern_leaf_blight": treatment(
        "Corn Northern Leaf Blight", "Fungal disease", "Exserohilum turcicum",
        ["Moderate temperatures of 65–80°F", "Prolonged leaf wetness or dew", "Susceptible hybrids and infected corn residue"],
        ["Apply fungicide from late vegetative to early reproductive stage (VT–R1) on susceptible hybrids with good yield potential.", "Remove and destroy heavily blighted plant debris.", "Improve field airflow with wider row spacing where feasible."],
        ["Plant hybrids carrying Ht resistance genes such as Ht1, Ht2, Ht3, or HtN.", "Rotate away from corn for at least one year.", "Use tillage to break down infected residue.", "Avoid dense planting that traps humidity in the canopy."],
    ),
    "lemon__anthracnose": treatment(
        "Lemon Anthracnose", "Fungal disease", "Colletotrichum gloeosporioides",
        ["Prolonged wet or foggy weather", "Stressed trees and weakened twigs", "Old or dead wood"],
        ["Prune dead, dying, and weakened twigs.", "Apply copper-based fungicide during extended wet periods.", "Remove and destroy fallen infected leaves and fruit."],
        ["Maintain tree vigor through proper irrigation and fertilization.", "Avoid mechanical wounding of fruit and branches.", "Prune for good air circulation and light penetration."],
    ),
    "lemon__bacterial_blight": treatment(
        "Lemon Bacterial Blight", "Bacterial disease", "Xanthomonas species",
        ["Wind-driven rain and splashing irrigation water", "Infected nursery stock", "Grove work while foliage is wet"],
        ["Apply copper-based bactericide sprays, especially before wet or rainy periods.", "Prune and destroy infected twigs and leaves.", "Disinfect pruning tools between cuts with diluted bleach or alcohol."],
        ["Plant certified disease-free nursery stock.", "Avoid overhead irrigation.", "Install windbreaks to reduce wind-driven rain.", "Avoid grove work while foliage is wet."],
    ),
    "lemon__citrus_canker": treatment(
        "Lemon Citrus Canker", "Bacterial disease", "Xanthomonas citri subsp. citri",
        ["Wind-driven rain", "Mechanical transmission through equipment", "Humid subtropical conditions"],
        ["There is no reliable cure once citrus canker is established in a tree.", "Copper bactericide sprays can reduce new lesion formation and slow spread.", "Remove and destroy severely infected plant material.", "Commercial groves may be subject to quarantine or eradication protocols; check local authority guidance."],
        ["Plant only certified disease-free stock.", "Use windbreaks.", "Sanitize tools and equipment between trees or groves.", "Do not move plant material from infected areas to clean areas.", "Use preventive copper sprays during high-risk wet and windy periods."],
        "Containment Steps",
    ),
    "lemon__curl_virus": treatment(
        "Lemon Curl Virus", "Possible viral disease or pest damage", "Requires local confirmation",
        ["Possible viral infection", "Sap-sucking mite or aphid feeding", "Aphid or whitefly vector activity"],
        ["If viral infection is confirmed, remove and destroy infected plants because there is no cure.", "Control aphid and whitefly vectors with neem oil or insecticidal soap."],
        ["Use certified virus-free planting material.", "Control sap-sucking insect vectors proactively.", "Disinfect tools between healthy and symptomatic plants.", "Remove and destroy infected trees early."],
        "Confirmation & Control Steps", "Citrus leaf curling can be viral or caused by sap-sucking pests. Confirm the cause locally before treatment.",
    ),
    "lemon__deficiency_leaf": treatment(
        "Lemon Deficiency Leaf", "Nutrient disorder", "Commonly nitrogen, zinc, or iron deficiency",
        ["Insufficient soil nutrients", "High soil pH blocking micronutrient uptake", "Inconsistent irrigation"],
        ["Get a soil or leaf-tissue test to identify the deficient nutrient.", "Apply a targeted correction such as chelated zinc or iron, or balanced NPK.", "Adjust soil pH if it is blocking nutrient uptake."],
        ["Test soil regularly and follow a scheduled fertilization program.", "Maintain soil pH between 6.0 and 7.5.", "Provide adequate, consistent irrigation."],
        "Correction Steps",
    ),
    "lemon__dry_leaf": treatment(
        "Lemon Dry Leaf", "Environmental or physiological disorder", "Water or heat stress",
        ["Insufficient or inconsistent irrigation", "Extreme heat and excessive sunlight", "Root problems that mimic drought stress"],
        ["Adjust irrigation frequency and volume to match weather conditions.", "Provide temporary shade during extreme heat.", "Apply mulch around the root zone."],
        ["Maintain consistent irrigation, especially during fruit development.", "Check for root problems such as root rot.", "Avoid allowing soil to dry out fully between waterings."],
        "Correction Steps",
    ),
    "lemon__sooty_mould": treatment(
        "Lemon Sooty Mould", "Secondary fungal growth", "Grows on honeydew from sap-sucking insects",
        ["Aphids, scale, whiteflies, or mealybugs", "Honeydew accumulation on leaves", "Ants protecting honeydew-producing pests"],
        ["Control the underlying insect pest first with neem oil or horticultural soap.", "After pests are controlled, wash mold from leaves with mild soap and water.", "Repeat insect treatment as needed."],
        ["Scout regularly for sap-sucking insects.", "Encourage natural predators such as ladybugs and lacewings.", "Control ant populations that protect honeydew-producing pests."],
        "Pest Control Steps",
    ),
    "lemon__spider_mites": treatment(
        "Lemon Spider Mites", "Pest", "Spider mites",
        ["Hot, dry, and dusty conditions", "Loss of natural predatory mites", "Excessive nitrogen fertilization"],
        ["Apply a miticide or acaricide according to its label.", "Use horticultural oil to suffocate mites and eggs.", "Increase humidity around the plant."],
        ["Wash dust from foliage periodically.", "Conserve natural predatory mites and avoid broad-spectrum insecticides.", "Avoid excessive nitrogen fertilization."],
        "Pest Control Steps",
    ),
    "pepper__bacterial_spot": treatment(
        "Bell Pepper Bacterial Spot", "Bacterial disease", "Xanthomonas euvesicatoria / X. campestris pv. vesicatoria",
        ["Infected seed or plant material", "Splashing water and overhead irrigation", "Field work while foliage is wet"],
        ["Apply copper-based bactericide, often combined with mancozeb to slow resistance development.", "Remove and destroy severely infected plants.", "Avoid working in the field while foliage is wet."],
        ["Use certified disease-free or hot-water-treated seed.", "Avoid overhead irrigation.", "Rotate away from tomato, pepper, and eggplant for 2–3 years.", "Disinfect stakes and tools between uses."],
    ),
    "tomato__bacterial_spot": treatment(
        "Tomato Bacterial Spot", "Bacterial disease", "Xanthomonas species",
        ["Infected seed or plant material", "Splashing water and overhead irrigation", "Field work while foliage is wet"],
        ["Apply copper-based bactericide, often combined with mancozeb to slow resistance development.", "Remove and destroy severely infected plants.", "Avoid working in the field while foliage is wet."],
        ["Use certified disease-free or hot-water-treated seed.", "Avoid overhead irrigation.", "Rotate away from tomato, pepper, and eggplant for 2–3 years.", "Disinfect stakes and tools between uses."],
    ),
    "tomato__early_blight": treatment(
        "Tomato Early Blight", "Fungal disease", "Alternaria solani",
        ["Infected crop debris", "Soil splash onto lower leaves", "Wet foliage and limited airflow"],
        ["Apply chlorothalonil or copper-based fungicide at the first sign of lesions and repeat per label interval.", "Remove and destroy infected lower leaves.", "Stake plants to improve airflow around foliage."],
        ["Rotate away from solanaceous crops for at least two years.", "Mulch to prevent soil splash.", "Water at the base and avoid wetting foliage.", "Use resistant varieties where available.", "Maintain adequate plant spacing."],
    ),
    "tomato__late_blight": treatment(
        "Tomato Late Blight", "Oomycete disease", "Phytophthora infestans",
        ["Cool, wet weather", "Prolonged leaf wetness", "Poor airflow and drainage"],
        ["Apply fungicide immediately at the first symptoms; a protectant and systemic combination is most effective.", "Remove and destroy infected plants immediately; do not compost them.", "In severe outbreaks, destroy the affected planting to reduce regional spread."],
        ["Use resistant varieties.", "Avoid overhead watering.", "Ensure good airflow and field drainage.", "Monitor regional late-blight forecasts.", "Avoid working in fields while foliage is wet."],
    ),
    "tomato__leaf_mold": treatment(
        "Tomato Leaf Mold", "Fungal disease", "Fulvia fulva (Passalora fulva)",
        ["Greenhouse or covered growing conditions", "Relative humidity around 85% or higher", "Dense planting and wet foliage"],
        ["Increase ventilation to reduce humidity.", "Apply a fungicide labeled for leaf mold.", "Remove and destroy infected leaves."],
        ["Increase plant spacing, especially under cover.", "Avoid overhead watering or misting.", "Use resistant varieties.", "Keep relative humidity below about 85% where possible."],
    ),
    "tomato__mosaic_virus": treatment(
        "Tomato Mosaic Virus", "Viral disease", "Tomato Mosaic Virus / Tobacco Mosaic Virus",
        ["Infected seed or plant material", "Contact through hands and contaminated tools", "Tobacco products carrying stable virus particles"],
        ["There is no cure; remove and destroy infected plants immediately.", "Wash hands and disinfect tools after handling infected plants."],
        ["Use certified virus-free or resistant seed varieties.", "Avoid handling tobacco products near plants.", "Disinfect tools between plants.", "Control aphids that vector related viruses."],
        "Containment Steps",
    ),
    "tomato__septoria_leaf_spot": treatment(
        "Tomato Septoria Leaf Spot", "Fungal disease", "Septoria lycopersici",
        ["Infected plant debris", "Soil splash onto lower foliage", "Wet leaves and limited airflow"],
        ["Remove infected lower leaves during the growing season.", "Apply fixed-copper or synthetic fungicide early when symptoms first appear.", "Remove all infected plant debris at the end of the season."],
        ["Stake and mulch plants.", "Improve airflow through spacing and sucker removal.", "Water at the base and avoid wetting foliage.", "Do not save seed from infected plants.", "Rotate crops."],
    ),
    "tomato__spider_mites_two_spotted_spider_mite": treatment(
        "Tomato Spider Mites / Two-Spotted Spider Mite", "Pest", "Tetranychus urticae",
        ["Hot and dry weather", "Drought-stressed plants", "Loss of natural predatory mites"],
        ["Apply a miticide or insecticidal soap according to its label.", "Increase humidity or spray plants with water to dislodge mites."],
        ["Avoid drought stress.", "Conserve predatory mites and avoid overusing broad-spectrum insecticides.", "Monitor plants closely during hot weather."],
        "Pest Control Steps",
    ),
    "tomato__target_spot": treatment(
        "Tomato Target Spot", "Fungal disease", "Corynespora cassiicola",
        ["Infected crop debris", "Wet foliage or overhead irrigation", "Dense planting with poor airflow"],
        ["Apply a fungicide used for early-blight management at the first sign of lesions.", "Remove severely infected leaves.", "Improve airflow around plants."],
        ["Rotate crops.", "Avoid overhead irrigation.", "Maintain adequate plant spacing.", "Use resistant varieties where available.", "Manage plant debris after harvest."],
    ),
    "tomato__yellow_leaf_curl_virus": treatment(
        "Tomato Yellow Leaf Curl Virus", "Viral disease", "TYLCV, transmitted by whiteflies",
        ["Whitefly vector activity", "Nearby infected fields or weed hosts", "Unprotected seedlings and young plants"],
        ["There is no cure; remove and destroy infected plants promptly.", "Aggressively control whiteflies with labeled insecticides, reflective mulch, or yellow sticky traps."],
        ["Use TYLCV-resistant varieties.", "Control whiteflies before symptoms appear.", "Use insect-proof netting on seedlings and young plants.", "Avoid planting near infected fields.", "Remove weed hosts near the planting area."],
        "Containment Steps",
    ),
}


def healthy_record(class_key: str) -> dict:
    crop = class_key.split("__", 1)[0].replace("pepper", "bell pepper").title()
    return treatment(
        f"{crop} Healthy",
        "Healthy",
        "No disease agent detected",
        ["The model did not detect a disease pattern in this image."],
        ["No disease treatment is recommended from this result.", "Continue routine crop monitoring and rescan if symptoms develop."],
        ["Use balanced irrigation and fertilization.", "Inspect leaves regularly for new spots, discoloration, pests, or curling.", "Keep tools clean and remove damaged plant material promptly."],
        "Care Steps",
        "A healthy prediction does not rule out every crop problem, especially when model confidence is low.",
    )


def get_treatment(class_key: str) -> dict | None:
    found = TREATMENTS.get(class_key)
    if found is None and class_key.endswith("__healthy"):
        found = healthy_record(class_key)
    if found is None:
        return None
    return {"class_key": class_key, **found}
