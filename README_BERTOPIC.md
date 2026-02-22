# BERTopic kurulumu (Windows)

Python **3.14** için hdbscan'ın hazır wheel'i yok; kurulum kaynak koddan derleme yapmaya çalışıyor ve C++ Build Tools ile bile başarısız olabiliyor.

**Çözüm:** Proje için **Python 3.11 veya 3.12** kullanın. Bu sürümlerde hdbscan için Windows wheel var, derleme gerekmez.

## Adımlar

### 1. Python 3.11 veya 3.12 kurun
- https://www.python.org/downloads/ — **3.11.x** veya **3.12.x** indirin
- Kurulumda **"Add Python to PATH"** işaretli olsun

### 2. Proje klasöründe sanal ortam oluşturun

**Windows (CMD veya PowerShell):**
```cmd
cd c:\Users\User\Documents\GitHub\urticaria-clone
py -3.11 -m venv .venv
.venv\Scripts\activate
```

veya Python 3.11 doğrudan PATH'teyse:
```cmd
python3.11 -m venv .venv
.venv\Scripts\activate
```

**Git Bash:**
```bash
cd ~/Documents/GitHub/urticaria-clone
py -3.11 -m venv .venv
source .venv/Scripts/activate
```

### 3. BERTopic ve bağımlılıkları kurun
```bash
python -m pip install --upgrade pip
python -m pip install bertopic sentence-transformers pandas matplotlib scikit-learn
```

### 4. Topic analizini çalıştırın
```bash
python topic_analysis_keywords.py
```

---

**Not:** `py -3.11` çalışmazsa (Python 3.11 yüklü değilse), önce 3.11 veya 3.12’yi indirip kurun; sonra aynı adımları o sürümle tekrarlayın.
