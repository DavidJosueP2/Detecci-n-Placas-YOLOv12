import re


NUMBER_TO_LETTER = {
    "0": "D",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "Y",
    "8": "B",
    "9": "G",
}

LETTER_TO_NUMBER = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "Y": "7",
    "B": "8",
}

PROVINCES = {
    "A": "Azuay",
    "B": "Bolivar",
    "C": "Carchi",
    "E": "Esmeraldas",
    "G": "Guayas",
    "H": "Chimborazo",
    "I": "Imbabura",
    "J": "Santo Domingo de los Tsachilas",
    "K": "Sucumbios",
    "L": "Loja",
    "M": "Manabi",
    "N": "Napo",
    "O": "El Oro",
    "P": "Pichincha",
    "Q": "Orellana",
    "R": "Los Rios",
    "S": "Pastaza",
    "T": "Tungurahua",
    "U": "Canar",
    "V": "Morona Santiago",
    "W": "Galapagos",
    "X": "Cotopaxi",
    "Y": "Santa Elena",
    "Z": "Zamora Chinchipe",
}


def postprocess_ecuador_plate(text, confidence=0.0, characters=None):
    text_original = str(text or "").upper()
    spatial = _spatial_text(characters or [])
    layout = spatial["layout"]
    text_for_rules = spatial["text"] or text_original
    text_clean = _clean_text(text_for_rules)

    if layout in {"moto_dos_lineas", "placa_dos_lineas"}:
        corrected, changes = _correct_two_line_plate(text_clean, spatial["rows"])
    else:
        corrected, changes = _correct_standard_plate(text_clean)

    valid = _is_valid_standard(corrected) if layout == "una_linea" else _is_valid_two_line(spatial["rows"])
    province = PROVINCES.get(corrected[:1])
    corrected_flag = corrected != text_clean

    return {
        "texto_original": text_original,
        "texto_limpio": text_clean,
        "texto_corregido": corrected,
        "confianza_original": float(confidence or 0.0),
        "corregida": corrected_flag,
        "cambios": changes,
        "cumple_formato_ecuador": valid,
        "provincia": province,
        "layout": layout,
        "validacion_espacial": spatial["used_spatial"],
    }


def _clean_text(text):
    return "".join(char for char in str(text or "").upper() if char.isalnum())


def _correct_standard_plate(text):
    trimmed = text[:7]
    changes = []
    corrected = []
    for index, char in enumerate(trimmed):
        new_char = char
        reason = None
        if index < 3 and char.isdigit() and char in NUMBER_TO_LETTER:
            new_char = NUMBER_TO_LETTER[char]
            reason = "Las tres primeras posiciones deben ser letras"
        elif index >= 3 and char.isalpha() and char in LETTER_TO_NUMBER:
            new_char = LETTER_TO_NUMBER[char]
            reason = "Desde la posicion 3 deben ser numeros"

        if reason:
            changes.append(
                {
                    "posicion": index,
                    "valor_original": char,
                    "valor_corregido": new_char,
                    "motivo": reason,
                }
            )
        corrected.append(new_char)
    return "".join(corrected), changes


def _correct_two_line_plate(text, rows):
    if not rows:
        return _correct_standard_plate(text)

    changes = []
    corrected_rows = []
    for row_index, row in enumerate(rows[:2]):
        row_chars = []
        for char_index, item in enumerate(row):
            char = _clean_text(item.get("value", ""))[:1]
            if not char:
                continue
            new_char = char
            reason = None
            if row_index == 0 and char.isdigit() and char in NUMBER_TO_LETTER:
                new_char = NUMBER_TO_LETTER[char]
                reason = "Fila superior de placa en dos lineas debe contener letras"
            elif row_index >= 1 and char.isalpha() and char in LETTER_TO_NUMBER:
                new_char = LETTER_TO_NUMBER[char]
                reason = "Fila inferior de placa en dos lineas debe contener numeros"
            if reason:
                changes.append(
                    {
                        "posicion": len("".join(corrected_rows)) + char_index,
                        "valor_original": char,
                        "valor_corregido": new_char,
                        "motivo": reason,
                    }
                )
            row_chars.append(new_char)
        corrected_rows.append("".join(row_chars))
    return "".join(corrected_rows)[:7], changes


def _is_valid_standard(text):
    return bool(re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", text or ""))


def _is_valid_two_line(rows):
    if len(rows) < 2:
        return False
    upper = "".join(_clean_text(item.get("value", ""))[:1] for item in rows[0])
    lower = "".join(_clean_text(item.get("value", ""))[:1] for item in rows[1])
    return bool(upper) and upper.isalpha() and bool(lower) and lower.isdigit()


def _spatial_text(characters):
    usable = [
        item
        for item in characters
        if item.get("value") and item.get("center_y") is not None and item.get("center_x") is not None
    ]
    if len(usable) < 4:
        return {
            "text": "",
            "rows": [],
            "layout": "una_linea",
            "used_spatial": False,
        }

    rows = _group_rows(usable)
    if len(rows) < 2:
        ordered = sorted(usable, key=lambda item: item.get("center_x", 0.0))
        return {
            "text": "".join(str(item.get("value", "")).upper()[:1] for item in ordered),
            "rows": [ordered],
            "layout": "una_linea",
            "used_spatial": True,
        }

    rows = rows[:2]
    text = "".join(
        str(item.get("value", "")).upper()[:1]
        for row in rows
        for item in row
    )
    upper = "".join(str(item.get("value", "")).upper()[:1] for item in rows[0])
    lower = "".join(str(item.get("value", "")).upper()[:1] for item in rows[1])
    layout = "moto_dos_lineas" if len(upper) <= 3 and len(lower) >= 2 else "placa_dos_lineas"
    return {
        "text": text,
        "rows": rows,
        "layout": layout,
        "used_spatial": True,
    }


def _group_rows(characters):
    ordered = sorted(characters, key=lambda item: item.get("center_y", 0.0))
    heights = [
        max(1.0, float(item.get("height", 0.0) or 0.0))
        for item in ordered
    ]
    median_height = sorted(heights)[len(heights) // 2]
    threshold = max(8.0, median_height * 0.65)
    rows = []
    for item in ordered:
        y = float(item.get("center_y", 0.0))
        target = None
        for row in rows:
            row_y = sum(float(current.get("center_y", 0.0)) for current in row) / len(row)
            if abs(y - row_y) <= threshold:
                target = row
                break
        if target is None:
            rows.append([item])
        else:
            target.append(item)

    rows.sort(key=lambda row: sum(float(item.get("center_y", 0.0)) for item in row) / len(row))
    return [sorted(row, key=lambda item: item.get("center_x", 0.0)) for row in rows]
