# Kurulum Talimatları — Urticaria YouTube İnfodemiyoloji Veri Toplama

Bu dosya, `datacollection.py` kodunu çalıştırmadan **önce** yapmanız gereken adımları açıklar.

---

## 1. Python Ortamı

- **Python 3.8+** yüklü olmalı
- Terminalde kontrol: `python --version` veya `python3 --version`

---

## 2. Bağımlılıkları Yükleme

Proje klasöründe:

```bash
pip install -r requirements.txt
```

veya tek tek:

```bash
pip install google-api-python-client pandas tqdm
```

---

## 3. Google Cloud Projesi ve YouTube API Key Oluşturma

### 3.1 Google Cloud Console'a Giriş

1. [Google Cloud Console](https://console.cloud.google.com/) adresine gidin
2. Google hesabınızla giriş yapın

### 3.2 Yeni Proje Oluşturma

1. Üst menüden **"Select a project"** → **"New Project"** tıklayın
2. Proje adı girin (örn: `urticaria-youtube-infodemiology`)
3. **Create** tıklayın

### 3.3 YouTube Data API v3'ü Etkinleştirme

1. Sol menüden **APIs & Services** → **Library** gidin
2. Arama kutusuna **"YouTube Data API v3"** yazın
3. **YouTube Data API v3** seçin → **Enable** tıklayın

### 3.4 API Key Oluşturma

1. Sol menüden **APIs & Services** → **Credentials** gidin
2. **+ Create Credentials** → **API key** seçin
3. Oluşan anahtarı kopyalayın
4. (Önerilir) **Edit API key** ile kısıtlama ekleyin:
   - **Application restrictions**: IP addresses veya HTTP referrers (web için)
   - **API restrictions**: Sadece **YouTube Data API v3** seçin

### 3.5 API Key'i Projede Kullanma

İki seçenek:

**Seçenek A — Ortam değişkeni (önerilen):**

Windows PowerShell:
```powershell
$env:YOUTUBE_API_KEY = "AIza...your-key-here"
```

Windows CMD:
```cmd
set YOUTUBE_API_KEY=AIza...your-key-here
```

**Seçenek B — Dosya ile:**

Proje klasöründe `config.py` oluşturun:
```python
YOUTUBE_API_KEY = "AIza...your-key-here"
```

`config.py` dosyasını **`.gitignore`** ekleyerek Git'e yüklemekten kaçının. Örnek: `config.example.py` oluşturup `YOUTUBE_API_KEY = ""` bırakın, kopyalayıp `config.py` yapın ve key'i girin.

---

## 4. API Kota Limitleri (Önemli)

YouTube Data API günlük **10.000 birim** kota verir:

| İşlem                    | Kota maliyeti  |
|--------------------------|----------------|
| `search.list` (her sayfa) | 100 birim      |
| `videos.list` (detay)     | 1 birim        |
| `commentThreads.list`     | 1 birim        |

- 8 anahtar kelime × ~2–3 sayfa ≈ 2.400 birim (video arama)
- Her video için yorum çekmek: video başına yaklaşık 7–10 birim (350 yorum ≈ 7 sayfa)
- Örnek: 200 video × 8 birim ≈ 1.600 birim

**Toplam:** İlk gün için yaklaşık 4.000–5.000 birim kullanılabilir. Daha fazla video için günlerce çalıştırmanız veya [quota artışı](https://support.google.com/googleapi/answer/7035613) talep etmeniz gerekebilir.

---

## 5. Çalıştırma Sırası

### Adım 1: Ham video listesini oluştur

```bash
python datacollection.py --step videos
```

Çıktı: `videos_raw.csv`

### Adım 2: Kural tabanlı otomatik filtre

Tüm videoları manuel incelemek yerine önce otomatik filtre uygulanır (~70–80% elenir):

```bash
python datacollection.py --step filter
```

Çıktılar:
- `videos_prefiltered.csv` — Tüm videolar + `prefilter_eligible`, `auto_exclusion_reason`
- `videos_shortlist.csv` — Sadece elenenlerin dışında kalanlar (~400–500 video)

**Filtre kuralları (metodoloji B2):**
- **Dahil:** Title veya description’da `urticaria` OR `hives` OR `angioedema` geçmeli
- **Hariç:** Eczema, dermatitis, drug rash/SJS/TEN, psoriasis, rosacea, viral exanthem, insect/bedbug, veterinary vb.

### Adım 3: Manuel eleme (sadece shortlist)

1. `videos_shortlist.csv` dosyasını açın (~400–500 satır)
2. Her satırı inceleyip `eligible_video` sütununa **1** (dahil) veya **0** (hariç) yazın
3. Hariç tuttuğunuz videolar için `exclusion_reason` sütununu doldurun
4. Sonucu `videos_clean.csv` olarak kaydedin

### Adım 4: Yorumları çek

```bash
python datacollection.py --step comments
```

Bu adım `videos_clean.csv` dosyasını okur ve sadece `eligible_video == 1` olan videolar için yorumları indirir.

Çıktılar:
- `comments_raw.csv`
- `comments_deduplicated.csv` (video başına max 350 yorum + kullanıcı başına ilk yorum)

---

## 6. Sorun Giderme

| Sorun | Olası çözüm |
|-------|--------------|
| `API key not found` | `YOUTUBE_API_KEY` ortam değişkeni veya `config.py` kontrol edin |
| `quotaExceeded` | Günlük kota dolmuştur; ertesi gün tekrar deneyin |
| `commentsDisabled` | Bazı videolarda yorum kapalıdır; bu videolar otomatik atlanır |
| `forbidden` / 403 | API etkinleştirildiğinden ve key'in doğru olduğundan emin olun |

---

## 7. Özet Checklist

- [ ] Python 3.8+ yüklü
- [ ] `pip install -r requirements.txt` çalıştırıldı
- [ ] Google Cloud projesi oluşturuldu
- [ ] YouTube Data API v3 etkinleştirildi
- [ ] API key oluşturuldu ve `YOUTUBE_API_KEY` veya `config.py` ile ayarlandı
- [ ] `python datacollection.py --step videos` ile `videos_raw.csv` üretildi
- [ ] `python datacollection.py --step filter` ile shortlist oluşturuldu
- [ ] Manuel eleme yapılıp (sadece shortlist üzerinde) `videos_clean.csv` hazırlandı
- [ ] `python datacollection.py --step comments` ile yorumlar çekildi
