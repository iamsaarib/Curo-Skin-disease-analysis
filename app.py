import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# Page configuration
st.set_page_config(page_title="Curo 🌿 - Skin Disease Classifier", layout="centered")

# Sidebar
st.sidebar.title("Why Curo?")
st.sidebar.info(
    "Curo is your AI-powered skin health companion 🧠🌿.\n\n"
    "It helps you quickly identify possible skin conditions by analyzing uploaded images. "
    "Whether you're dealing with irritation, rashes, or unexplained marks — Curo gives you a starting point, "
    "along with likely causes, treatment suggestions, and soothing home remedies.\n\n"
    "This is not a replacement for a doctor, but it can guide you to seek help early and stay informed."
)


# Main Header
st.markdown("<h1 style='text-align: center;'>🌿 Curo – Skin Disease Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an image of the affected skin area, and the app will classify the disease and provide analysis.")

st.markdown("---")  # horizontal line

# Load model & processor
@st.cache_resource
def load_model():
    repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
    processor = AutoImageProcessor.from_pretrained(repo_name)
    model = AutoModelForImageClassification.from_pretrained(repo_name)
    return model, processor

model, processor = load_model()
# Ensure the model is in evaluation mode
model.eval()

# Define the class names
class_names = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa',
    'Hailey-Hailey Disease', 'Herpes Simplex', 'Impetigo', 'Larva Migrans',
    'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid', 'Lichen Planus',
    'Lupus Erythematosus Chronicus Discoides', 'Melanoma', 'Molluscum Contagiosum',
    'Mycosis Fungoides', 'Neurofibromatosis', 'Papilomatosis Confluentes And Reticulate',
    'Pediculosis Capitis', 'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis',
    'Tinea Corporis', 'Tinea Nigra', 'Tungiasis', 'actinic keratosis', 'dermatofibroma',
    'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma',
    'vascular lesion'
]

# Dictionary of disease reasons, treatments, and remedies (truncated for brevity)
disease_analysis = {
    "Basal Cell Carcinoma": {
        "reason": "Caused by prolonged exposure to ultraviolet (UV) radiation from sunlight or tanning beds, as well as genetic predisposition.",
        "treatment": "Surgical removal, radiation therapy, or topical treatments.",
        "home_remedy": "Apply aloe vera gel to soothe the skin and use green tea extracts for antioxidant benefits."
    },
    "Darier_s Disease": {
        "reason": "A rare genetic disorder caused by mutations in the ATP2A2 gene, leading to issues with skin cell adhesion.",
        "treatment": "Retinoids, moisturizers, and sun protection.",
        "home_remedy": "Use oatmeal baths to relieve irritation and avoid tight clothing to prevent friction."
    },
    "Epidermolysis Bullosa Pruriginosa": {
        "reason": "A rare genetic disorder causing skin fragility and blistering.",
        "treatment": "Wound care, pain management, and avoiding trauma to the skin.",
        "home_remedy": "Apply coconut oil for soothing and keep the skin hydrated with gentle moisturizers."
    },
    "Hailey-Hailey Disease": {
        "reason": "A genetic disorder caused by mutations in the ATP2C1 gene, leading to improper skin cell cohesion.",
        "treatment": "Topical steroids, antibiotics, and avoiding friction or heat.",
        "home_remedy": "Cool compresses and aloe vera gel to relieve discomfort."
    },
    "Herpes Simplex": {
        "reason": "Caused by the herpes simplex virus (HSV), typically transmitted through direct contact or saliva.",
        "treatment": "Antiviral medications like acyclovir or valacyclovir.",
        "home_remedy": "Apply cold compresses or honey to reduce pain and inflammation."
    },
    "Impetigo": {
        "reason": "A bacterial infection caused by Staphylococcus aureus or Streptococcus pyogenes.",
        "treatment": "Topical or oral antibiotics.",
        "home_remedy": "Clean the affected area with diluted vinegar and apply tea tree oil for antimicrobial effects."
    },
    "Larva Migrans": {
        "reason": "Caused by parasitic hookworms that infect the skin, usually through contaminated soil.",
        "treatment": "Anti-parasitic medications like albendazole or ivermectin.",
        "home_remedy": "Soak the affected area in warm water and keep the skin clean."
    },
    "Leprosy Borderline": {
        "reason": "Caused by the bacterium Mycobacterium leprae, typically spread through prolonged close contact.",
        "treatment": "Multi-drug therapy including rifampin, dapsone, and clofazimine.",
        "home_remedy": "Boost immune health with a balanced diet rich in vitamin C and antioxidants."
    },
    "Leprosy Lepromatous": {
        "reason": "A severe form of leprosy caused by Mycobacterium leprae, associated with immune system dysfunction.",
        "treatment": "Long-term multi-drug therapy.",
        "home_remedy": "Include turmeric in the diet for its anti-inflammatory properties."
    },
    "Leprosy Tuberculoid": {
        "reason": "A milder form of leprosy caused by Mycobacterium leprae, with localized skin lesions.",
        "treatment": "Multi-drug therapy including rifampin and dapsone.",
        "home_remedy": "Maintain proper hygiene and support the immune system with vitamin-rich foods."
    },
    "Lichen Planus": {
        "reason": "Thought to be an autoimmune condition triggered by infections, medications, or stress.",
        "treatment": "Topical steroids, antihistamines, and light therapy.",
        "home_remedy": "Apply aloe vera gel to soothe the skin and use turmeric paste for inflammation."
    },
    "Lupus Erythematosus Chronicus Discoides": {
        "reason": "An autoimmune condition triggered by sunlight exposure and genetic factors.",
        "treatment": "Sun protection, topical steroids, and antimalarial drugs.",
        "home_remedy": "Use calendula cream for soothing and avoid sun exposure."
    },
    "Melanoma": {
        "reason": "Caused by mutations in melanocytes, often due to excessive UV radiation exposure and genetic factors.",
        "treatment": "Surgical excision, immunotherapy, or targeted therapy.",
        "home_remedy": "Apply green tea extracts for antioxidant support and avoid sun exposure."
    },
    "Molluscum Contagiosum": {
        "reason": "A viral infection caused by the molluscum contagiosum virus, spread through skin-to-skin contact or contaminated objects.",
        "treatment": "Cryotherapy, topical treatments, or curettage.",
        "home_remedy": "Apply apple cider vinegar as a natural antiseptic."
    },
    "Mycosis Fungoides": {
        "reason": "A type of cutaneous T-cell lymphoma with unknown exact causes but potentially linked to immune dysfunction.",
        "treatment": "Phototherapy, topical treatments, or systemic medications.",
        "home_remedy": "Use coconut oil for hydration and gentle skin care products."
    },
    "Neurofibromatosis": {
        "reason": "A genetic disorder caused by mutations in the NF1 or NF2 genes, leading to benign tumor growth.",
        "treatment": "Surgical removal of tumors and symptom management.",
        "home_remedy": "Maintain a healthy diet and avoid skin irritation."
    },
    "Papilomatosis Confluentes And Reticulate": {
        "reason": "Often associated with genetic factors or chronic irritation.",
        "treatment": "Symptomatic treatment and monitoring.",
        "home_remedy": "Apply aloe vera gel to soothe irritation."
    },
    "Pediculosis Capitis": {
        "reason": "Caused by infestation with head lice (Pediculus humanus capitis), transmitted through close contact.",
        "treatment": "Topical insecticides or manual removal.",
        "home_remedy": "Use a mixture of coconut oil and tea tree oil to remove lice."
    },
    "Pityriasis Rosea": {
        "reason": "Likely caused by viral infections, though the exact virus is unknown.",
        "treatment": "Symptomatic relief with antihistamines or topical treatments.",
        "home_remedy": "Apply calamine lotion for relief and take lukewarm oatmeal baths."
    },
    "Porokeratosis Actinic": {
        "reason": "Caused by prolonged UV exposure or genetic factors, leading to abnormal keratinization.",
        "treatment": "Cryotherapy, topical treatments, or laser therapy.",
        "home_remedy": "Use sunscreen regularly and apply aloe vera for soothing."
    },
    "Psoriasis": {
        "reason": "An autoimmune condition triggered by stress, infections, or genetic predisposition.",
        "treatment": "Topical steroids, phototherapy, or systemic medications.",
        "home_remedy": "Apply coconut oil for moisture and use oatmeal baths for relief."
    },
    "Tinea Corporis": {
        "reason": "A fungal infection caused by dermatophytes, often transmitted through contact with infected individuals or surfaces.",
        "treatment": "Topical or oral antifungal medications.",
        "home_remedy": "Apply tea tree oil to the affected area and keep the skin dry."
    },
    "Tinea Nigra": {
        "reason": "A rare fungal infection caused by Hortaea werneckii, often contracted in tropical regions.",
        "treatment": "Topical antifungal treatments.",
        "home_remedy": "Use apple cider vinegar for cleansing and antifungal effects."
    },
    "Tungiasis": {
        "reason": "Caused by infestation of the skin by the sand flea (Tunga penetrans), often from walking barefoot.",
        "treatment": "Manual removal of fleas and wound care.",
        "home_remedy": "Apply antiseptic and keep the area clean."
    },
    "actinic keratosis": {
        "reason": "Caused by prolonged sun exposure leading to abnormal skin cell changes.",
        "treatment": "Cryotherapy, topical treatments, or laser therapy.",
        "home_remedy": "Use sunscreen and aloe vera for soothing."
    },
    "dermatofibroma": {
        "reason": "Likely caused by minor skin injuries or insect bites, leading to localized fibroblast proliferation.",
        "treatment": "Observation or surgical removal if necessary.",
        "home_remedy": "Apply turmeric paste for natural anti-inflammatory benefits."
    },
    "nevus": {
        "reason": "Usually congenital or caused by genetic mutations in skin cells (melanocytes).",
        "treatment": "Monitoring or surgical removal if changes are observed.",
        "home_remedy": "Apply coconut oil for hydration."
    },
    "pigmented benign keratosis": {
        "reason": "Often caused by aging and prolonged UV exposure.",
        "treatment": "Observation or cryotherapy for cosmetic reasons.",
        "home_remedy": "Use green tea extracts for antioxidant benefits."
    },
    "seborrheic keratosis": {
        "reason": "Caused by aging and genetic factors, with no known environmental triggers.",
        "treatment": "Cryotherapy, curettage, or observation.",
        "home_remedy": "Apply coconut oil for skin hydration."
    },
    "squamous cell carcinoma": {
        "reason": "Caused by prolonged UV exposure, chemical exposure, or chronic skin irritation.",
        "treatment": "Surgical removal, radiation therapy, or topical treatments.",
        "home_remedy": "Use aloe vera for soothing and apply green tea extracts."
    },
    "vascular lesion": {
        "reason": "Caused by abnormal growth or formation of blood vessels, often due to genetic or developmental factors.",
        "treatment": "Laser treatment or surgical intervention.",
        "home_remedy": "Apply cold compresses and use calendula cream for soothing."
    }
}

# Function to classify the image
def classify_image(image):
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
    predicted_label = class_names[predicted_class_idx]
    return predicted_label

# File uploader for user image
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Analyze the image
    with st.spinner("Analyzing the image..."):
        predicted_label = classify_image(image)
        reason = disease_analysis.get(predicted_label, {}).get("reason", "Reason unknown.")
        treatment = disease_analysis.get(predicted_label, {}).get("treatment", "Consult a dermatologist.")
        home_remedy = disease_analysis.get(predicted_label, {}).get("home_remedy", "No specific home remedies available.")

    # Display the results
    st.success("Analysis Complete!")
    st.write(f"**Classification**: {predicted_label}")
    st.write(f"**Reason**: {reason}")
    st.write(f"**Treatment**: {treatment}")
    st.write(f"**Home Remedy**: {home_remedy}")
    st.write("**Note:** Please consult a doctor for final recommendations and a detailed treatment plan.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.9em; color: grey;'>"
    "Developed by Saarib · Powered by Hugging Face · Still a work in progress 🚧"
    "</div>",
    unsafe_allow_html=True
)

