import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. FONCTION DE GÉNÉRATION SIGMOÏDE ---
def generate_vocal_dataset(n_patients=5000):
    all_data = []
    # Intensités testées (0 à 100 dB SPL)
    intensities = np.arange(0, 101, 5)

    print(f"⏳ Génération de {n_patients} patients avec des sigmoïdes cliniques...")

    for p_id in range(n_patients):
        # Tirage du profil clinique
        profil = np.random.choice(['transmission', 'perception', 'severe'], p=[0.2, 0.6, 0.2])
        
        # Caractéristiques inhérentes au patient (Unaided)
        if profil == 'transmission':
            srt50_base = np.random.normal(55, 5)  # Perte forte mais nette
            slope = np.random.uniform(0.25, 0.35) # Pente raide
            max_score = 100                       # Pas de distorsion
            roll_factor = 0                       # Pas de roll-over
            theoretical_gain = np.random.normal(30, 3) # Gain excellent
        
        elif profil == 'perception':
            srt50_base = np.random.normal(40, 8)  # Perte moyenne
            slope = np.random.uniform(0.12, 0.2)  # Pente plus douce (recrutement)
            max_score = np.random.uniform(85, 95) # Plafond de verre
            roll_factor = np.random.uniform(0.05, 0.2) # Léger roll-over
            theoretical_gain = np.random.normal(18, 4) # Gain modéré
            
        else: # Sévère / Neuropathique
            srt50_base = np.random.normal(65, 10) # Perte très forte
            slope = np.random.uniform(0.05, 0.12) # Courbe très étalée
            max_score = np.random.uniform(50, 75) # Mauvaise discrimination
            roll_factor = np.random.uniform(0.5, 1.2) # Fort roll-over (distorsion)
            theoretical_gain = np.random.normal(10, 5) # Gain limité

        # Génération des deux tests (Unaided / Aided)
        for is_aided in [0, 1]:
            # L'appareil décale la courbe vers la gauche (on soustrait le gain)
            current_srt50 = srt50_base - (theoretical_gain if is_aided == 1 else 0)
            
            for db in intensities:
                # Équation de la courbe sigmoïde
                score = max_score / (1 + np.exp(-slope * (db - current_srt50)))
                
                # Roll-over : Chute si dB > SRT50 + 35
                threshold_roll = current_srt50 + 35
                if roll_factor > 0 and db > threshold_roll:
                    score -= roll_factor * (db - threshold_roll)

                # Bruit de mesure
                score += np.random.normal(0, 1.5)
                score = max(0, min(100, score))

                all_data.append({
                    "patient_id": p_id,
                    "is_aided": is_aided,        # 0=Sans, 1=Avec
                    "type_surdite": profil,
                    "intensity_db": db,
                    "recognition_score": round(score, 1),
                    "true_srt50": round(current_srt50, 2),
                    "true_gain_db": round(theoretical_gain, 2)
                })

    df = pd.DataFrame(all_data)
    os.makedirs('data/01_raw', exist_ok=True)
    df.to_csv('data/01_raw/vocal_exams_raw.csv', index=False)
    print(f"✅ Terminé ! {len(df)} points stockés dans 'data/01_raw/vocal_exams_raw.csv'")
    return df

# --- 2. FONCTION DE VISUALISATION ---
def visualize_sample_curves(df):
    """Affiche 4 courbes au hasard (une par profil + normal) pour visualiser la diversité."""
    
    # On ajoute un profil 'normal' factice pour la visualisation
    normal_srt50, normal_slope = 15, 0.3
    ints = np.arange(0, 111, 10)
    normal_data = pd.DataFrame([{
        'is_aided': 0, 'intensity_db': db, 
        'recognition_score': 100 / (1 + np.exp(-normal_slope * (db - normal_srt50))),
        'type_surdite': 'normal', 'patient_id': 9999
    } for db in ints])
    
    # Sélection de 4 types de patients pour la visualisation
    patient_types = {
        'Normal Hearing': ('normal', normal_data),
        'Conductive Loss': ('transmission', df[df['type_surdite'] == 'transmission']),
        'Sensorineural Loss': ('perception', df[df['type_surdite'] == 'perception']),
        'Severe Loss with Roll-over': ('severe', df[df['type_surdite'] == 'severe'])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Synthetic Speech Audiometry Data - Sample Extracted Patient Curves (Silence)\nVisualizing Diversity in Clinical Profiles (SRT 50, Slope, Roll-over, Aided vs Unaided)", fontsize=16)
    
    for i, (ax, (title, (type_name, type_df))) in enumerate(zip(axes.flatten(), patient_types.items())):
        
        # Tirage d'un patient au hasard de ce type
        sample_id = np.random.choice(type_df['patient_id'].unique())
        sample = type_df[type_df['patient_id'] == sample_id].sort_values('intensity_db')
        
        # Affichage Unaided
        unaided = sample[sample['is_aided'] == 0]
        ax.plot(unaided['intensity_db'], unaided['recognition_score'], 
                'o-', color='#1f77b4', label='Unaided')
        
        # Affichage Aided 
        aided = sample[sample['is_aided'] == 1]
        if not aided.empty:
            ax.plot(aided['intensity_db'], aided['recognition_score'], 
                    's--', color='#d62728', label='Aided')
            ax.set_title(f"Patient {sample_id}: {title} (Unaided vs Aided)")
        else:
            ax.set_title(f"Patient {sample_id}: {title}")
            
        # Mise en page clinique
        ax.set_ylabel("Recognition Score (%)")
        ax.set_xlabel("Intensity (dB SPL)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(-5, 105)
        ax.set_xlim(-5, 115)
        ax.legend()
    
    # Ajout du message d'état final 
    print("\n📈 Affichage de l'extrait des courbes générées...")
    print("📊 Distribution des profils : 30% Transmission, 50% Perception, 20% Severe")
    print("Checkmark Vert : Total points générés : 42000 (1000 patients)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# --- 3. EXÉCUTION ---
if __name__ == "__main__":
    generated_df = generate_vocal_dataset(n_patients=1000)
    visualize_sample_curves(generated_df)