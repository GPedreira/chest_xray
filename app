import streamlit as st
import gdown
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import io
import pydicom
import os
import pandas as pd

# ============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# ============================================================================

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(LABELS)
IMG_SIZE = 500
MC_LOOPS = 20 # Número de passadas para calcular incerteza (MC Dropout)

# ============================================================================
# 2. DEFINIÇÃO DA ARQUITETURA (Idêntica ao Treino)
# ============================================================================

class ChestXrayModel(nn.Module):
    def __init__(self, use_gender=True):
        super().__init__()
        self.use_gender = use_gender
        # Carrega estrutura sem pesos pré-treinados (vamos carregar os nossos)
        self.backbone = models.efficientnet_b3(weights=None)
        n_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_feat + (1 if use_gender else 0), 512),
            nn.ReLU(), 
            nn.Dropout(0.5), # Essencial para MC Dropout
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.5), # Essencial para MC Dropout
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x, gender=None):
        feat = self.backbone(x)
        if self.use_gender and gender is not None:
            # Garante dimensão correta
            gender = gender.unsqueeze(1) if gender.dim() == 1 else gender
            feat = torch.cat([feat, gender], dim=1)
        return self.classifier(feat)

# ============================================================================
# 3. FUNÇÕES AUXILIARES
# ============================================================================

def enable_dropout(model):
    """Força o Dropout a ficar ativo mesmo no modo de inferência (eval)"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def get_valid_transforms(img_size=500):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.224]),
        ToTensorV2()
    ])

@st.cache_resource
def load_model_from_drive(file_id):
    """Baixa e carrega o modelo, com cache para não baixar toda hora"""
    output_path = "model_checkpoint.pth"
    
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            with st.spinner("Baixando modelo do servidor... (Isso acontece apenas uma vez)"):
                gdown.download(url, output_path, quiet=False)
        except Exception as e:
            st.error(f"Erro ao baixar modelo: {e}")
            return None

    try:
        model = ChestXrayModel(use_gender=True)
        # Map location cpu garante que rode mesmo sem GPU
        checkpoint = torch.load(output_path, map_location=torch.device('cpu'))
        
        # Trata se o checkpoint salvou o dicionário inteiro ou só o state_dict
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval() # Trava Batch Norm, mas vamos destravar o Dropout depois
        return model
    except Exception as e:
        st.error(f"Erro ao carregar pesos do modelo: {e}")
        return None

def process_image(uploaded_file):
    """Lê DICOM ou Imagem comum e prepara tensor"""
    try:
        file_bytes = uploaded_file.read()
        
        if uploaded_file.name.lower().endswith('.dcm'):
            dcm = pydicom.dcmread(io.BytesIO(file_bytes))
            img_array = dcm.pixel_array
            # Normaliza DICOM para 0-255
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            pil_image = Image.fromarray(img_array)
        else:
            pil_image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            img_array = np.array(pil_image)

        transform = get_valid_transforms(IMG_SIZE)
        img_tensor = transform(image=img_array)['image'].unsqueeze(0)
        
        return img_tensor, pil_image
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")
        return None, None

# ============================================================================
# 4. APLICAÇÃO STREAMLIT
# ============================================================================

def main():
    st.set_page_config(page_title="AI Chest X-Ray Analyzer", page_icon="🩻", layout="wide")
    
    # --- HEADER ---
    st.title("🩻 Diagnóstico Assistido por IA (Raio-X Tórax)")
    st.markdown("""
    Esta ferramenta utiliza Deep Learning (**EfficientNet-B3**) com **Análise de Incerteza Bayesiana** para detectar 14 patologias torácicas comuns.
    """)
    
    # --- SIDEBAR (Configurações) ---
    with st.sidebar:
        st.header("⚙️ Configurações")
        # INPUT DO ID DO DRIVE (Para facilitar sua vida)
        # Substitua este valor padrão pelo ID do seu arquivo model_BEST_F1_FULL.pth
        #https://drive.google.com/uc?id=16YdJi8bAMXMYaD-z1suNFX6nA2PwlDJp
        default_id = "https://drive.google.com/uc?id=16YdJi8bAMXMYaD-z1suNFX6nA2PwlDJp" 
        drive_id = st.text_input("ID do Modelo no Google Drive:", value=default_id)
        
        st.info("O modelo realiza múltiplas passadas (MC Dropout) para estimar a confiança do diagnóstico.")

    # --- CARREGAMENTO DO MODELO ---
    if len(drive_id) > 10:
        model = load_model_from_drive(drive_id)
    else:
        st.warning("👈 Por favor, insira o ID do arquivo do Google Drive na barra lateral.")
        st.stop()

    if not model:
        st.stop()

    # --- INTERFACE PRINCIPAL ---
    col_upload, col_result = st.columns([1, 1.5])

    with col_upload:
        st.subheader("1. Upload do Exame")
        uploaded_file = st.file_uploader("Envie Raio-X (DICOM, PNG, JPG)", type=['png', 'jpg', 'jpeg', 'dcm'])
        
        # Gênero (Como o modelo foi treinado com isso, precisamos pedir, ou fixar neutro)
        gender_opt = st.radio("Gênero do Paciente (Opcional)", ["Desconhecido/Neutro", "Masculino", "Feminino"], horizontal=True)
        if gender_opt == "Masculino": gender_val = 1.0
        elif gender_opt == "Feminino": gender_val = 0.0
        else: gender_val = 0.5

    if uploaded_file and model:
        img_tensor, pil_image = process_image(uploaded_file)
        
        with col_upload:
            st.image(pil_image, caption="Imagem Original", use_column_width=True)

        if st.button("🔍 Analisar Raio-X", type="primary"):
            with col_result:
                with st.spinner(f"Rodando {MC_LOOPS} simulações clínicas para análise de incerteza..."):
                    
                    # --- INFERÊNCIA MC DROPOUT ---
                    model.eval()
                    enable_dropout(model) # Ativa incerteza
                    
                    preds = []
                    gender_tensor = torch.tensor([gender_val], dtype=torch.float32)
                    
                    with torch.no_grad():
                        for _ in range(MC_LOOPS):
                            out = model(img_tensor, gender_tensor)
                            preds.append(torch.sigmoid(out).cpu().numpy()[0])
                    
                    preds = np.array(preds)
                    mean_probs = preds.mean(axis=0)
                    std_devs = preds.std(axis=0)

                    # --- LÓGICA DO LAUDO ---
                    # Probabilidade de ser anormal = Maior probabilidade de qualquer doença
                    max_disease_prob = np.max(mean_probs)
                    prob_normal = 1.0 - max_disease_prob
                    
                    st.subheader("2. Laudo da IA")
                    
                    # --- STATUS GERAL ---
                    if prob_normal > 0.5:
                        st.success(f"✅ **APARÊNCIA NORMAL** (Confiança: {prob_normal*100:.1f}%)")
                    else:
                        st.error(f"🚨 **SINAIS DE ANORMALIDADE DETECTADOS** (Probabilidade: {max_disease_prob*100:.1f}%)")

                    st.divider()

                    # --- DETALHES POR DOENÇA ---
                    findings = []
                    for i in range(NUM_CLASSES):
                        prob = mean_probs[i]
                        unc = std_devs[i]
                        
                        # Classificação de Risco e Confiança
                        if prob > 0.50: risk = "ALTO"
                        elif prob > 0.20: risk = "MODERADO"
                        elif prob > 0.05: risk = "BAIXO"
                        else: continue # Não mostra riscos insignificantes
                        
                        if unc < 0.05: conf_level = "Alta"
                        elif unc < 0.15: conf_level = "Média"
                        else: conf_level = "Baixa (Incerto)"
                        
                        findings.append({
                            "Patologia": LABELS[i],
                            "Probabilidade": prob,
                            "Risco": risk,
                            "Incerteza (±)": unc,
                            "Confiança IA": conf_level
                        })
                    
                    # Ordenar e Exibir
                    if findings:
                        df_res = pd.DataFrame(findings).sort_values(by="Probabilidade", ascending=False)
                        
                        for _, row in df_res.iterrows():
                            pat = row['Patologia']
                            prob = row['Probabilidade']
                            unc = row['Incerteza (±)']
                            
                            # Cor da barra dependendo do risco
                            if prob > 0.5: color = "red"
                            elif prob > 0.25: color = "orange"
                            else: color = "yellow"
                            
                            col_txt, col_bar = st.columns([2, 3])
                            with col_txt:
                                st.markdown(f"**{pat}**")
                                st.caption(f"Prob: {prob*100:.1f}% ±{unc*100:.1f}%")
                            with col_bar:
                                st.progress(min(prob, 1.0), text=row['Confiança IA'])
                    else:
                        st.info("Nenhum achado patológico significativo (>5%).")

if __name__ == "__main__":
    main()
