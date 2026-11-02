import streamlit as st
st.markdown("""
<link rel="manifest" href="/manifest.json">
<script>
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js');
}
</script>
""", unsafe_allow_html=True)
st.info("للتثبيت كتطبيق: افتح الموقع من الموبايل -> 3 نقط -> Add to Home Screen")
