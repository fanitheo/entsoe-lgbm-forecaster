# ENTSO-E LGBM Forecaster
*🇬🇧 [English Version](#english-version) | 🇬🇷 [Ελληνική Έκδοση](#ελληνική-έκδοση)*

---

## 🇬🇧 English Version

This repository contains a pre-trained LightGBM (Machine Learning) model designed to forecast energy data using the ENTSO-E API. The system fetches real-time Day-Ahead data and generates predictions automatically.

The project is written in **Python** and is fully **Dockerized**, ensuring a consistent and isolated execution environment across any operating system.

### ⚙️ How It Works

1. **Data Fetching:** The script is designed to run daily **after 15:00 (CET/CEST)**, as this is when the ENTSO-E Day-Ahead data for the following day becomes available.
2. **German Market Mechanism:** A specialized module is integrated into the `main.py` script to handle the fetching and preprocessing of data specifically for the German energy market.
3. **Inference:** Instead of training from scratch, the system loads a highly optimized, pre-trained LightGBM model to process the real-time data and generate forecasts instantly.
4. **Export:** The predicted values are automatically saved as a `.csv` file in the designated output directory.

### 📂 Repository Structure

* `main.py`: The main execution script. **Note:** This script also dynamically handles the installation of required Python libraries, so no separate `requirements.txt` is needed.
* `entsoe_day_ahead_API.py`: The module responsible for interacting with the ENTSO-E REST API.
* `Dockerfile`: Instructions for building the isolated container environment.
* `[Pre-trained Model]`: The saved LightGBM model used for inference (e.g., `.pkl` or `.txt`).
* `Report.pdf`: Comprehensive documentation detailing the methodology, data processing, and model training.

### 🛠️ Prerequisites

To run this project, you only need to have **Docker** installed on your machine.
* [Get Docker](https://docs.docker.com/get-docker/)

### 🚀 Installation & Usage

Follow these steps in your terminal to generate today's forecast:

**1. Build the Docker Image:**
Build the environment using the provided Dockerfile.
```bash
docker build --no-cache -t lgbm-forecaster .
```

**2. Run the Container:**
Execute the model to fetch data and generate predictions.
```bash
docker run --rm -v "${PWD}:/app/data" -v "${PWD}:/app/output" lgbm-forecaster
```
*(The `-v` flags ensure that the script can access the local pre-trained model and save the resulting `.csv` forecast file back to your machine).*

---

## 🇬🇷 Ελληνική Έκδοση

Αυτό το project περιέχει ένα προεκπαιδευμένο μοντέλο μηχανικής μάθησης (LightGBM), το οποίο χρησιμοποιεί δεδομένα από το ENTSO-E API για να κάνει προβλέψεις ενέργειας. Το σύστημα ανακτά δεδομένα Day-Ahead σε πραγματικό χρόνο και παράγει τις προβλέψεις αυτόματα.

Είναι γραμμένο σε **Python** και πλήρως **Dockerized**, ώστε να εκτελείται εύκολα σε οποιονδήποτε υπολογιστή.

### ⚙️ Πώς Λειτουργεί

1. **Λήψη Δεδομένων:** Το script πρέπει να εκτελείται καθημερινά **μετά τις 15:00**, καθώς τότε γίνονται διαθέσιμες οι τιμές της επόμενης ημέρας από το ENTSO-E.
2. **Μηχανισμός Γερμανίας:** Στο αρχείο `main.py` υπάρχει ενσωματωμένος ένας ειδικός μηχανισμός για τη λήψη και διαχείριση δεδομένων ειδικά από τη γερμανική αγορά.
3. **Πρόβλεψη (Inference):** Το σύστημα φορτώνει το ήδη εκπαιδευμένο μοντέλο LightGBM για να κάνει την πρόβλεψη άμεσα, χωρίς να απαιτείται επανεκπαίδευση.
4. **Εξαγωγή:** Οι προβλεπόμενες τιμές αποθηκεύονται αυτόματα σε ένα ξεχωριστό αρχείο `.csv` στον φάκελο εξόδου.

### 📂 Δομή του Repository

* `main.py`: Ο κύριος κώδικας του συστήματος. **Σημείωση:** Το αρχείο αυτό αναλαμβάνει εσωτερικά και την εγκατάσταση των απαραίτητων βιβλιοθηκών Python, οπότε δεν απαιτείται αρχείο `requirements.txt`.
* `entsoe_day_ahead_API.py`: Το αρχείο που διαχειρίζεται την επικοινωνία με το ENTSO-E API.
* `Dockerfile`: Οι οδηγίες για τη δημιουργία του περιβάλλοντος Docker.
* `[Προεκπαιδευμένο Μοντέλο]`: Το έτοιμο μοντέλο LightGBM που χρησιμοποιείται για τις προβλέψεις.
* `Αναφορά.pdf`: Η πλήρης αναφορά υλοποίησης με λεπτομέρειες για τη μεθοδολογία και την εκπαίδευση.

### 🛠️ Προαπαιτούμενα

Το μόνο που χρειάζεται να έχετε εγκατεστημένο στον υπολογιστή σας είναι το **Docker**.
* [Λήψη Docker](https://docs.docker.com/get-docker/)

### 🚀 Οδηγίες Εγκατάστασης & Εκτέλεσης

Εκτελέστε τις παρακάτω εντολές στο τερματικό σας:

**1. Χτίσιμο (Build) του Image:**
```bash
docker build --no-cache -t lgbm-forecaster .
```

**2. Εκτέλεση (Run) του Container:**
```bash
docker run --rm -v "${PWD}:/app/data" -v "${PWD}:/app/output" lgbm-forecaster
```
*(Οι παράμετροι `-v` εξασφαλίζουν ότι το παραγόμενο αρχείο `.csv` με τις προβλέψεις θα αποθηκευτεί τοπικά στον υπολογιστή σας και ότι το script έχει πρόσβαση στο μοντέλο).*