import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import pandas as pd
import numpy as np
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import tensorflow as tf
from datetime import datetime

# Initialize tokenizer with specific parameters
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-base", 
                                          model_max_length=512,
                                          legacy=True)  # Add legacy=True
except Exception as e:
    st.error(f"Error loading tokenizer: {str(e)}")
  
def get_topic_unit_mapping():
    return {
        # PHYSICS UNITS
        "UNIT 1: Units and Measurements": [
            "units", "measurement", "SI Units", "dimension", "significant figures",
            "least count", "errors", "dimensional analysis", "physical quantities",
            "fundamental units", "derived units", "accuracy", "precision", "parallax",
            "vernier caliper", "screw gauge", "systematic error", "random error"
        ],
        
        "UNIT 2: Kinematics": [
            "motion", "velocity", "acceleration", "displacement", "frame of reference",
            "uniform motion", "non-uniform motion", "average speed", "instantaneous velocity",
            "projectile motion", "circular motion", "relative velocity", "position-time graph",
            "velocity-time graph", "free fall", "trajectory", "range", "time of flight",
            "angular velocity", "centripetal acceleration", "vector", "scalar"
        ],
        
        "UNIT 3: Laws of Motion": [
            "force", "newton", "inertia", "momentum", "impulse", "friction",
            "centripetal force", "banking of roads", "equilibrium", "concurrent forces",
            "free body diagram", "tension", "normal force", "applied force",
            "action reaction", "conservation of momentum", "collision", "thrust",
            "pseudo force", "centrifugal force", "static friction", "kinetic friction"
        ],
        
        "UNIT 4: Work, Energy and Power": [
            "work", "energy", "power", "kinetic energy", "potential energy",
            "conservation of energy", "collision", "work-energy theorem", "elastic collision",
            "inelastic collision", "gravitational potential energy", "spring constant",
            "elastic potential energy", "conservative force", "non-conservative force",
            "work done", "zero work", "positive work", "negative work", "instantaneous power"
        ],
        
        "UNIT 5: Rotational Motion": [
            "rotation", "torque", "angular momentum", "moment of inertia", "radius of gyration",
            "rolling motion", "rigid body", "center of mass", "angular velocity",
            "angular acceleration", "rotational kinetic energy", "angular displacement",
            "rolling without slipping", "pure rolling", "couple", "parallel axis theorem",
            "perpendicular axis theorem", "conservation of angular momentum"
        ],
        
        "UNIT 6: Gravitation": [
            "gravity", "gravitational field", "gravitational potential", "escape velocity",
            "orbital velocity", "satellite", "kepler laws", "planetary motion",
            "universal gravitational constant", "acceleration due to gravity", "weight",
            "gravitational force", "geostationary orbit", "polar satellite", "binding energy",
            "gravitational potential energy", "planetary orbits", "weightlessness"
        ],
        
        "UNIT 7: Properties of Solids and Liquids": [
            "elastic", "stress", "strain", "hooke", "young modulus", "bulk modulus",
            "pascal law", "pressure", "fluid", "viscosity", "surface tension", "bernoulli",
            "elasticity", "plastic behavior", "shear modulus", "hydraulic lift",
            "streamline flow", "turbulent flow", "reynolds number", "stokes law",
            "cohesive force", "adhesive force", "capillarity", "venturi effect"
        ],
        
        "UNIT 8: Thermodynamics": [
            "temperature", "heat", "thermal equilibrium", "zeroth law", "first law",
            "second law", "thermodynamic processes", "entropy", "heat engine",
            "carnot cycle", "efficiency", "refrigerator", "heat pump", "isothermal",
            "adiabatic", "isobaric", "isochoric", "internal energy", "work done",
            "heat capacity", "specific heat", "latent heat", "calorimetry"
        ],
        
        "UNIT 9: Kinetic Theory of Gases": [
            "kinetic theory", "ideal gas", "gas laws", "rms speed", "degrees of freedom",
            "mean free path", "avogadro number", "equipartition", "boyle law",
            "charles law", "gay lussac law", "dalton law", "maxwell distribution",
            "brownian motion", "real gas", "van der waals", "critical temperature",
            "mean kinetic energy", "pressure temperature relationship"
        ],
        
        "UNIT 10: Oscillations and Waves": [
            "oscillation", "periodic motion", "simple harmonic motion", "frequency",
            "time period", "amplitude", "phase", "wave motion", "standing waves", "beats",
            "resonance", "forced oscillation", "damped oscillation", "longitudinal wave",
            "transverse wave", "wave speed", "wavelength", "wave equation", "doppler effect",
            "sound waves", "intensity", "loudness", "pitch", "quality", "harmonics"
        ],
        
        "UNIT 11: Electrostatics": [
            "electric charge", "coulomb law", "electric field", "electric potential",
            "gauss law", "capacitance", "dielectric", "conductor", "insulator",
            "electrostatic force", "point charge", "dipole", "dipole moment",
            "electric flux", "field lines", "equipotential surface", "electrostatic energy",
            "parallel plate capacitor", "series combination", "parallel combination"
        ],
        
        # CHEMISTRY UNITS
        "UNIT 1: Some Basic Concepts in Chemistry": [
            "matter", "atom", "molecule", "mole concept", "atomic mass", "molecular mass",
            "formula mass", "stoichiometry", "limiting reagent", "empirical formula",
            "molecular formula", "percentage composition", "molarity", "molality",
            "normality", "atomic number", "mass number", "isotopes", "isobars", "isotones",
            "significant figures", "scientific notation", "dimensional analysis"
        ],
        
        "UNIT 2: Atomic Structure": [
            "bohr model", "quantum numbers", "orbital", "electronic configuration",
            "aufbau principle", "hund rule", "pauli exclusion", "photoelectric effect",
            "wave particle duality", "de broglie wavelength", "heisenberg uncertainty",
            "quantum mechanical model", "s p d f orbitals", "electron spin",
            "electromagnetic spectrum", "atomic spectra", "hydrogen spectrum",
            "rutherford model", "thomson model", "dual nature", "quantum theory"
        ],
        
        "UNIT 3: Chemical Bonding and Molecular Structure": [
            "chemical bond", "ionic bond", "covalent bond", "electronegativity",
            "VSEPR theory", "molecular orbital", "hybridization", "sigma bond",
            "pi bond", "bond order", "resonance", "valence bond theory",
            "molecular geometry", "dipole moment", "hydrogen bonding",
            "metallic bonding", "van der waals forces", "coordinate bond",
            "lattice energy", "bond angle", "bond length", "orbital overlap"
        ],
        
        "UNIT 4: Chemical Thermodynamics": [
            "system", "surroundings", "state functions", "enthalpy", "entropy",
            "gibbs energy", "first law", "second law", "third law", "heat capacity",
            "calorimetry", "hess law", "bond energy", "spontaneous process",
            "reversible process", "irreversible process", "free energy",
            "equilibrium constant", "thermochemical equations", "standard state",
            "heat of formation", "heat of combustion", "heat of neutralization"
        ],
        
        "UNIT 5: Solutions": [
            "solution", "solute", "solvent", "concentration", "molarity", "molality",
            "mole fraction", "henry law", "raoult law", "vapor pressure", "osmosis",
            "osmotic pressure", "colligative properties", "freezing point depression",
            "boiling point elevation", "van't hoff factor", "abnormal molecular mass",
            "reverse osmosis", "isotonic solution", "ideal solution", "non-ideal solution",
            "azeotropes", "solubility", "miscibility"
        ],
        
        "UNIT 6: Equilibrium": [
            "chemical equilibrium", "reversible reaction", "law of mass action",
            "equilibrium constant", "le chatelier principle", "common ion effect",
            "buffer solution", "ph", "poh", "ionic equilibrium", "solubility product",
            "degree of dissociation", "acid base equilibria", "hydrolysis constant",
            "henderson equation", "buffer capacity", "indicators", "ph titration",
            "weak acid", "weak base", "strong acid", "strong base", "salt hydrolysis"
        ],
        
        "UNIT 7: Redox Reactions and Electrochemistry": [
            "oxidation", "reduction", "redox reaction", "oxidation number",
            "electron transfer", "oxidizing agent", "reducing agent", "electrochemical cell",
            "electrolytic cell", "galvanic cell", "standard electrode potential",
            "nernst equation", "emf", "faraday laws", "conductance", "electrolysis",
            "corrosion", "battery", "fuel cell", "electrolyte", "anode", "cathode",
            "cell notation", "half cell", "salt bridge"
        ],
        
        "UNIT 8: Chemical Kinetics": [
            "reaction rate", "rate law", "rate constant", "order of reaction",
            "molecularity", "activation energy", "arrhenius equation", "catalyst",
            "rate determining step", "reaction mechanism", "elementary reaction",
            "complex reaction", "half life", "collision theory", "threshold energy",
            "temperature coefficient", "pseudo first order", "zero order", "first order",
            "second order", "integrated rate equation"
        ],
        
        "UNIT 9: Surface Chemistry": [
            "adsorption", "absorption", "sorption", "physisorption", "chemisorption",
            "adsorbate", "adsorbent", "desorption", "catalysis", "colloids",
            "emulsion", "gel", "sol", "tyndall effect", "brownian movement",
            "electrophoresis", "coagulation", "peptization", "micelle", "emulsifier",
            "heterogeneous catalysis", "homogeneous catalysis", "surface tension"
        ],
        
        "UNIT 10: Coordination Chemistry": [
            "coordination compound", "complex compound", "ligand", "coordination number",
            "central metal", "chelate", "monodentate", "bidentate", "polydentate",
            "iupac nomenclature", "isomerism", "geometrical isomerism", "optical isomerism",
            "crystal field theory", "color", "magnetic properties", "stability",
            "werner theory", "effective atomic number", "primary valence", "secondary valence"
        ],
        
        "UNIT 11: p-Block Elements": [
            "group 13", "group 14", "group 15", "group 16", "group 17", "group 18",
            "boron family", "carbon family", "nitrogen family", "oxygen family",
            "halogen family", "noble gases", "inert gases", "allotropes", "oxides",
            "hydrides", "oxyacids", "interhalogen compounds", "xenon compounds",
            "silicates", "silicones", "borax", "diborane"
        ],
        
        "UNIT 12: d and f Block Elements": [
            "transition elements", "inner transition elements", "lanthanoids",
            "actinoids", "d block", "f block", "electronic configuration",
            "oxidation states", "complex formation", "magnetic properties",
            "catalytic properties", "colored ions", "alloy formation",
            "interstitial compounds", "lanthanide contraction", "actinide contraction",
            "atomic radii", "ionization enthalpy", "stereochemistry"
        ],
        
        "UNIT 13: Organic Chemistry - Basic Principles": [
            "organic compounds", "carbon", "hybridization", "isomerism", "nomenclature",
            "functional groups", "homologous series", "resonance", "hyperconjugation",
            "inductive effect", "electromeric effect", "carbocation", "carbanion",
            "free radical", "nucleophile", "electrophile", "reaction mechanism",
            "substitution", "addition", "elimination", "rearrangement"
        ],
        
        "UNIT 14: Hydrocarbons": [
            "alkanes", "alkenes", "alkynes", "aromatics", "benzene", "nomenclature",
            "isomerism", "conformations", "preparation", "physical properties",
            "chemical properties", "markovnikov rule", "anti markovnikov rule",
            "wurtz reaction", "addition reaction", "elimination reaction",
            "aromatic substitution", "resonance structure", "huckel rule"
        ],
        
        "UNIT 15: Environmental Chemistry": [
            "pollution", "air pollution", "water pollution", "soil pollution",
            "greenhouse effect", "global warming", "ozone depletion", "acid rain",
            "smog", "photochemical smog", "particulate matter", "water treatment",
            "sewage treatment", "biodegradable", "non-biodegradable", "biomagnification",
            "green chemistry", "pesticides", "heavy metals"
        ],
        
        "UNIT 16: Polymers": [
            "polymer", "monomer", "polymerization", "addition polymerization",
            "condensation polymerization", "copolymer", "natural polymer",
            "synthetic polymer", "thermoplastic", "thermosetting", "elastomer",
            "fiber", "plastic", "rubber", "vulcanization", "biodegradable polymer",
            "molecular mass", "degree of polymerization"
        ],
        
        "UNIT 17: Biomolecules": [
            "carbohydrates", "proteins", "enzymes", "vitamins", "nucleic acids",
            "DNA", "RNA", "amino acids", "peptide bond", "glucose", "fructose",
            "primary structure", "secondary structure", "tertiary structure",
            "quaternary structure", "denaturation", "enzyme catalysis",
            "coenzyme", "metabolism", "biological functions"
        ],
        
        "UNIT 18: Chemistry in Everyday Life": [
            "drugs", "medicines", "chemicals", "food chemistry", "cleansing agents",
            "soaps", "detergents", "antiseptics", "disinfectants", "analgesics",
            "antipyretics", "antibiotics", "antacids", "food preservatives",
            "artificial sweeteners", "biodegradable detergents", "therapeutic effect",
            "side effects", "drug-drug interaction"
        ]
    }

def get_related_topics(unit, topic):
    mapping = get_topic_unit_mapping()
    if unit in mapping:
        keywords = mapping[unit]
        topic_lower = topic.lower()
        related = []
        for keyword in keywords:
            if keyword.lower() != topic_lower and (keyword.lower() in topic_lower or topic_lower in keyword.lower()):
                related.append(keyword)
        return related[:3]  # Return top 3 related topics
    return []

@st.cache_data
def load_and_organize_data():
    df = pd.read_csv('NCERT_dataset.csv', low_memory=False)
    
    # Filter only Physics and Chemistry
    df = df[df['subject'].isin(['Physics', 'Chemistry'])]
    
    def assign_unit(row):
        content = f"{row['Topic']} {row['Explanation']}".lower()
        subject = row['subject']
        mapping = get_topic_unit_mapping()
        
        max_matches = 0
        best_unit = None
        
        for unit, keywords in mapping.items():
            if ((subject == "Physics" and "UNIT" in unit) or 
                (subject == "Chemistry" and "UNIT" in unit)):
                matches = sum(1 for keyword in keywords if keyword.lower() in content)
                if matches > max_matches:
                    max_matches = matches
                    best_unit = unit
        
        return best_unit if best_unit else f"UNIT Other: {row['Topic']}"
    
    df['Unit'] = df.apply(assign_unit, axis=1)
    return df

def get_unit_prerequisites(unit):
    prerequisites = {
        # Physics prerequisites
        "UNIT 2: Kinematics": ["UNIT 1: Units and Measurements"],
        "UNIT 3: Laws of Motion": ["UNIT 1: Units and Measurements", "UNIT 2: Kinematics"],
        "UNIT 4: Work, Energy and Power": ["UNIT 2: Kinematics", "UNIT 3: Laws of Motion"],
        "UNIT 5: Rotational Motion": ["UNIT 3: Laws of Motion", "UNIT 4: Work, Energy and Power"],
        "UNIT 6: Gravitation": ["UNIT 1: Units and Measurements", "UNIT 3: Laws of Motion"],
        "UNIT 11: Electrostatics": ["UNIT 1: Units and Measurements", "UNIT 3: Laws of Motion"],
        
        # Chemistry prerequisites
        "UNIT 2: Atomic Structure": ["UNIT 1: Some Basic Concepts in Chemistry"],
        "UNIT 3: Chemical Bonding": ["UNIT 2: Atomic Structure"],
        "UNIT 4: Chemical Thermodynamics": ["UNIT 1: Some Basic Concepts in Chemistry"],
        "UNIT 7: Redox Reactions": ["UNIT 1: Some Basic Concepts in Chemistry", "UNIT 6: Equilibrium"],
        "UNIT 14: Organic Chemistry": ["UNIT 1: Some Basic Concepts in Chemistry", "UNIT 3: Chemical Bonding"]
    }
    return prerequisites.get(unit, [])


class QASystem:
    def __init__(self, model_path='my_qa_model_tf'):
        """Initialize the QA system with model and data."""
        try:
            # Print available files for debugging
            st.write(f"Loading model from: {model_path}")
            st.write("Available files:", os.listdir(model_path))
            
            # Initialize tokenizer and model directly from tf-base
            self.tokenizer = T5Tokenizer.from_pretrained(
                "t5-base",
                model_max_length=512,
                legacy=True
            )
            
            # Load model config first
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found at {config_path}")
            
            # Initialize model with TensorFlow
            self.model = TFT5ForConditionalGeneration.from_pretrained(
                "t5-base"  # First load base model
            )
            
            # Load your trained weights
            model_path = os.path.join(model_path, "tf_model.h5")
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found at {model_path}")
            
            # Load the weights
            self.model.load_weights(model_path)
            
            # Load and organize data
            self.df = load_and_organize_data()
            
            # Set up configuration parameters
            self.max_input_length = 512
            self.max_output_length = 128
            self.min_answer_length = 5
            self.temperature = 0.7
            self.num_beams = 4
            
            st.success("Model initialized successfully!")
            
        except Exception as e:
            st.error(f"Error initializing QA System: {str(e)}")
            st.error(f"Current directory: {os.getcwd()}")
            st.error(f"Files in current directory: {os.listdir('.')}")
            raise

    def get_units(self, subject):
        """Get all units for a given subject."""
        try:
            units = sorted(self.df[self.df['subject'] == subject]['Unit'].unique())
            if not units:
                raise ValueError(f"No units found for subject: {subject}")
            return units
        except Exception as e:
            st.error(f"Error getting units: {str(e)}")
            return []

    def get_topics_for_unit(self, subject, unit):
        """Get all topics for a given unit and subject."""
        try:
            unit_data = self.df[
                (self.df['subject'] == subject) & 
                (self.df['Unit'] == unit)
            ]
            topics = sorted(unit_data['Topic'].unique())
            if not topics:
                raise ValueError(f"No topics found for unit: {unit}")
            return topics
        except Exception as e:
            st.error(f"Error getting topics: {str(e)}")
            return []

    def get_unit_content(self, subject, unit):
        """Get concatenated content for a given unit."""
        try:
            unit_data = self.df[
                (self.df['subject'] == subject) & 
                (self.df['Unit'] == unit)
            ]
            if unit_data.empty:
                raise ValueError(f"No content found for unit: {unit}")
            return ' '.join(unit_data['Explanation'].dropna().tolist())
        except Exception as e:
            st.error(f"Error getting unit content: {str(e)}")
            return ""

    def get_topic_content(self, subject, unit, topic):
        """Get content for a specific topic."""
        try:
            topic_data = self.df[
                (self.df['subject'] == subject) & 
                (self.df['Unit'] == unit) & 
                (self.df['Topic'] == topic)
            ]
            if topic_data.empty:
                raise ValueError(f"No content found for topic: {topic}")
            return topic_data['Explanation'].iloc[0]
        except Exception as e:
            st.error(f"Error getting topic content: {str(e)}")
            return ""

    def preprocess_question(self, question):
        """Preprocess the question for better understanding."""
        try:
            # Remove extra whitespace
            question = ' '.join(question.split())
            
            # Add question mark if missing
            if not question.endswith('?'):
                question += '?'
            
            # Add common prefixes for better model understanding
            if not any(question.lower().startswith(prefix) for prefix in ['what', 'how', 'why', 'when', 'where', 'who', 'explain']):
                question = f"Explain {question}"
            
            return question
        except Exception as e:
            st.error(f"Error preprocessing question: {str(e)}")
            return question

    def generate_answer(self, context, question):
        """Generate an answer for the given question and context."""
        try:
            # Input validation
            if not context or not question:
                raise ValueError("Context and question cannot be empty")
            
            # Preprocess input
            input_text = f"question: {question} context: {context}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors='tf',  # Change to tf
                max_length=self.max_input_length,
                truncation=True,
                padding='max_length'
            )
            
            # Generate answer using TensorFlow
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_output_length,
                num_beams=4,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            # Decode answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return self.cleanup_answer(answer)

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "Sorry, there was an error generating the answer. Please try again."

    def calculate_confidence(self, answer):
        """Calculate confidence score for the generated answer."""
        try:
            # Simple heuristic based on answer length and complexity
            word_count = len(answer.split())
            sentence_count = len(answer.split('.'))
            
            # Calculate base confidence
            confidence = min(word_count / 50.0, 1.0)
            
            # Adjust for sentence complexity
            if sentence_count > 1:
                confidence *= min(sentence_count / 3.0, 1.0)
            
            return confidence
        except Exception as e:
            st.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def validate_answer(self, answer):
        """Validate the generated answer."""
        try:
            # Check minimum length
            if len(answer.split()) < self.min_answer_length:
                return False
            
            # Check for common error indicators
            error_indicators = ['error', 'sorry', 'apologize', 'couldn\'t', 'cannot']
            if any(indicator in answer.lower() for indicator in error_indicators):
                return False
            
            return True
        except Exception as e:
            st.error(f"Error validating answer: {str(e)}")
            return False

    def cleanup_answer(self, answer):
        """Clean up the generated answer."""
        try:
            # Remove extra whitespace
            answer = ' '.join(answer.split())
            
            # Ensure proper capitalization
            answer = answer[0].upper() + answer[1:]
            
            # Ensure proper punctuation
            if not answer.endswith(('.', '?', '!')):
                answer += '.'
            
            return answer
        except Exception as e:
            st.error(f"Error cleaning up answer: {str(e)}")
            return answer

    def save_to_history(self, subject, unit, topic, question, answer, difficulty):
        """Save QA interaction to history."""
        try:
            history_item = {
                'subject': subject,
                'unit': unit,
                'topic': topic,
                'question': question,
                'answer': answer,
                'difficulty': difficulty,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'confidence': f"{self.calculate_confidence(answer):.2%}"
            }
            return history_item
        except Exception as e:
            st.error(f"Error saving to history: {str(e)}")
            return None
        
def main():
    st.set_page_config(page_title="Physics & Chemistry QA System", page_icon="🔬", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .topic-header {
            color: #1E88E5;
            font-size: 20px;
            font-weight: bold;
        }
        .difficulty-badge-easy {
            color: green;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
        }
        .difficulty-badge-medium {
            color: orange;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
        }
        .difficulty-badge-hard {
            color: red;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
        }
        .answer-box {
            background-color: #1E1E1E;
            color: #FFFFFF;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border-left: 5px solid #1E88E5;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Physics & Chemistry Question Answering System")
    
    try:
        # Initialize QA system
        qa_system = QASystem()
        
        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'current_topic' not in st.session_state:
            st.session_state.current_topic = ''
        if 'question' not in st.session_state:
            st.session_state.question = ''
        if 'context' not in st.session_state:
            st.session_state.context = ''

        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            
            # Subject selection
            subject = st.radio("Choose subject:", 
                             ["Physics ⚡", "Chemistry ⚗️"])
            subject = subject.split()[0]  # Remove emoji
            
            # Unit selection
            st.subheader("Select Unit")
            search_unit = st.text_input("Search units...", "")
            units = sorted(qa_system.df[qa_system.df['subject'] == subject]['Unit'].unique())
            
            if search_unit:
                units = [u for u in units if search_unit.lower() in u.lower()]
            selected_unit = st.selectbox("Choose Unit:", units)

            # Prerequisites
            prerequisites = get_unit_prerequisites(selected_unit)
            if prerequisites:
                st.subheader("Prerequisites")
                for prereq in prerequisites:
                    st.info(f"📌 {prereq}")

        # Get unit data
        unit_data = qa_system.df[
            (qa_system.df['subject'] == subject) & 
            (qa_system.df['Unit'] == selected_unit)
        ]
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        # Content Column (Left)
        with col1:
            st.subheader(f"{selected_unit}")
            
            # Unit overview
            with st.expander("📖 Unit Overview", expanded=True):
                if not unit_data.empty:
                    st.write(unit_data['Explanation'].iloc[0])
                else:
                    st.write("No overview available")
            
            # Topics
            topics = sorted(unit_data['Topic'].unique())
            for topic in topics:
                with st.expander(f"📚 {topic}"):
                    topic_data = unit_data[unit_data['Topic'] == topic]
                    
                    if not topic_data.empty:
                        st.markdown("### Explanation")
                        st.write(topic_data['Explanation'].iloc[0])
                        
                        related = get_related_topics(selected_unit, topic)
                        if related:
                            st.markdown("### Related Topics")
                            for rel in related:
                                st.write(f"• {rel}")
                        
                        st.markdown("### Practice Questions")
                        for difficulty in ['Easy', 'Medium', 'Hard']:
                            diff_questions = topic_data[topic_data['Difficulty'] == difficulty]
                            if not diff_questions.empty:
                                st.markdown(f"#### {difficulty} Questions")
                                for idx, row in diff_questions.iterrows():
                                    question = row['Question']
                                    
                                    q_col1, q_col2 = st.columns([4, 1])
                                    with q_col1:
                                        st.write(f"Q: {question}")
                                    with q_col2:
                                        st.markdown(
                                            f"<div class='difficulty-badge-{difficulty.lower()}'>{difficulty}</div>",
                                            unsafe_allow_html=True
                                        )
                                    
                                    if st.button(f"Try this question", key=f"q_{topic}_{idx}"):
                                        st.session_state.question = question
                                        st.session_state.context = row['Explanation']
                                        st.session_state.current_topic = topic

        # Q&A Column (Right)
        with col2:
            st.subheader("Ask a Question")
            question = st.text_input("Enter your question:", 
                                   value=st.session_state.question)
            
            # Question templates
            if st.checkbox("Show question templates"):
                templates = [
                    "What is ...?",
                    "How does ... work?",
                    "Explain the concept of ...",
                    "Compare ... and ...",
                    "What are the applications of ...?"
                ]
                selected_template = st.selectbox("Choose a template:", templates)
                if selected_template:
                    st.session_state.question = selected_template
            
            # Answer generation
            if st.button("Get Answer", key="answer_button"):
                 if not question:
                   st.warning("Please enter a question.")
                 else:
                    with st.spinner("Generating answer..."):
                        try:
                            # Get context
                            context = st.session_state.get('context', 
                                                        qa_system.get_unit_content(subject, selected_unit))
                            
                            # Add topic-specific context
                            if st.session_state.current_topic:
                                topic_context = qa_system.get_topic_content(
                                    subject, selected_unit, st.session_state.current_topic)
                                if topic_context:
                                    context = f"{topic_context} {context}"
                
                            # Generate answer
                            answer = qa_system.generate_answer(context, question)
                            
                            # Always display the answer
                            st.markdown("### Answer")
                            st.markdown(f"""
                                <div class="answer-box">
                                    {answer}
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Calculate confidence
                            confidence = qa_system.calculate_confidence(answer)
                            st.progress(confidence)
                            st.write(f"Confidence: {confidence:.2%}")
                            
                            # Add to history
                            history_item = {
                                'subject': subject,
                                'unit': selected_unit,
                                'topic': st.session_state.current_topic,
                                'question': question,
                                'answer': answer,
                                'difficulty': topic_data.iloc[0]['Difficulty'] if not topic_data.empty else 'Medium',
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'confidence': f"{confidence:.2%}"
                            }
                            st.session_state.history.append(history_item)
                            
                            # Feedback buttons
                            fb_col1, fb_col2, fb_col3 = st.columns(3)
                            with fb_col1:
                                if st.button("👍 Helpful"):
                                    st.success("Thank you for your feedback!")
                            with fb_col2:
                                if st.button("👎 Not Helpful"):
                                    st.error("Sorry! We'll try to improve.")
                            with fb_col3:
                                if st.button("🔄 Try Again"):
                                    st.experimental_rerun()
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.info("Please try rephrasing your question or selecting a different topic.")

            # History section
            if st.session_state.history:
                st.subheader("Recent Questions")
                
                # History filters
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    filter_subject = st.multiselect("Filter by subject:", 
                                                  ["Physics", "Chemistry"])
                with filter_col2:
                    filter_difficulty = st.multiselect("Filter by difficulty:", 
                                                     ["Easy", "Medium", "Hard"])
                
                # Apply filters
                history = st.session_state.history
                if filter_subject:
                    history = [h for h in history if h['subject'] in filter_subject]
                if filter_difficulty:
                    history = [h for h in history if h['difficulty'] in filter_difficulty]
                
                # Display history
                for item in reversed(history[-5:]):
                    with st.expander(f"{item['subject']} - {item['unit']} ({item['timestamp']})"):
                        st.write(f"**Topic:** {item['topic']}")
                        st.markdown(
                            f"<div class='difficulty-badge-{item['difficulty'].lower()}'>{item['difficulty']}</div>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Q:** {item['question']}")
                        st.write(f"**A:** {item['answer']}")

            # Footer
            st.markdown("---")
            with st.expander("📚 Study Tips and Resources"):
                tip_col1, tip_col2 = st.columns(2)
                with tip_col1:
                    st.markdown("### Study Tips")
                    st.write("""
                    1. Start with the basic concepts in each unit
                    2. Practice questions from all difficulty levels
                    3. Review related topics to build connections
                    4. Use the question history to track your progress
                    5. Try to explain concepts in your own words
                    """)
                with tip_col2:
                    st.markdown("### Additional Resources")
                    st.write("""
                    • NCERT Textbooks
                    • Previous Year Questions
                    • Video Lectures
                    • Practice Problems
                    • Interactive Simulations
                    """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()





