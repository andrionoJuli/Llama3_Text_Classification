sex_drive_labels = {
    1: ["Tidak Berhubungan Seks", "Not having sex"],
    2: ["Seks Yang Terlindungi", "Protected sex"],
    3: ["Seks Yang Tidak Terlindungi", "Unprotected sex"],
    4: ["Hasrat Seksual Yang Tinggi", "High sex drive"],
    5: ["Masturbasi", "Masturbating"]
}

mood_labels = {
    1: ["Energik", "Energetic"],
    2: ["Senang Bahagia", "Happy"],
    3: ["Sedih", "Sad"],
    4: ["Depresi", "Depressed"],
    5: ["Bingung", "Confused"],
    6: ["Lincah", "Agile"],
    7: ["Gelisah", "Nervous"],
    8: ["Apatis","Apathetic"],
    9: ["Kesal", "Annoyed"],
    10: ["Merasa Bersalah", "Feel guilty"],
    11: ["Kritis Terhadap Diri Sendiri","Self critical"],
    12: ["Perubahan Suasana Hati", "Mood swings"]
}

symptoms_labels = {
    1: ["Baik-Baik Saja", "Just fine"],
    2: ["Keram Perut", "Stomach cramps"],
    3: ["Sakit Kepala", "Headache"],
    4: ["Jerawat", "Acne"],
    5: ["Mual", "Nauseous"],
    6: ["Kelelahan", "Fatigue"],
    7: ["Insomnia", "Insomnia"],
    8: ["Sembelit", "Constipation"],
    9: ["Diare", "Diarrhea"],
    0: ["Kaki Bengkak", "Swollen foot"],
    11: ["Kembung", "Bloated"],
    12: ["Mengidam", "Craving"],
    13: ["Nyeri Payudara", "Breast pain"],
    14: ["Sakit Punggung", "Back pain"]
}

vaginal_discharge_labels = {
    1: ["Tidak Ada Sekresi", "No Secretion"],
    2: ["Bercak", "Spotting"],
    3: ["Lengket", "Sticky"],
    4: ["Putih Telur", "Egg Whites"],
    5: ["Berair", "Watery"],
    6: ["Tidak Biasa", "Unusual"],
    7: ["Putih Kekuningan", "Yellowish White"]
}

def parse_labels(output):
    labels = {}
    lines = output.strip().split('\n')
    for line in lines:
        if ':' in line:
            parts = line.split(': ', 1)
            if len(parts) == 2:
                key, value = parts
                labels[key.strip()] = value.strip()
            else:
                labels[line.strip()] = None
        else:
            labels[line.strip()] = None
    return labels

def get_label_info(label, label_dict):
    for key, values in label_dict.items():
        for value in values:
            if value.lower() == label.lower():
                return key, values
    return None, None

def map_labels_to_ids(parsed_labels):
    label_dicts = {
        "Sex drive": sex_drive_labels,
        "Mood": mood_labels,
        "Symptoms": symptoms_labels,
        "Vaginal discharge": vaginal_discharge_labels
    }

    label_ids = {}

    for label_type, label_dict in label_dicts.items():
        labels = parsed_labels.get(label_type, "").split(", ")
        if labels == [""]:
            # Handle empty labels
            label_ids[label_type] = []
            continue

        label_ids[label_type] = []
        for label in labels:
            label_id, label_values = get_label_info(label, label_dict)
            if label_id is not None:
                item = {"id": label_id}
                if len(label_values) == 2:
                    item["name"], item["name_en"] = label_values
                else:
                    item["name"], item["name_en"] = label_values[0], label_values[1:]
                label_ids[label_type].append(item)
            else:
                # Handle labels not found
                label_ids[label_type].append({"id": None, "name": label, "name_en": label})

    return label_ids