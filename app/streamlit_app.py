import glob,sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from inference.generate import generate
st.set_page_config(page_title='MiniGPT',layout='centered')
st.title('MiniGPT')
files=glob.glob('checkpoints/*.pt'); checkpoint=st.selectbox('Checkpoint',files)
prompt=st.text_area('Prompt'); tokens=st.number_input('Maximum new tokens',1,1000,200); temperature=st.slider('Temperature',.1,2.,.8); top_k=st.number_input('Top-K (0 disables)',0,500,40); top_p=st.slider('Top-P (1 disables)',.1,1.,.95)
if st.button('Generate',type='primary',disabled=not files):
 with st.spinner('Generating...'): st.text_area('Generated output',generate(prompt,checkpoint,tokens,temperature,top_k or None,top_p if top_p<1 else None),height=300)
