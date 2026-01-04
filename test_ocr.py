import re

class MockService:
    def correct_sri_lankan_text(self, text):
        """
        Robust correction for Sri Lankan plates.
        Handles:
        - Provinces: WP, CP, SP, NP, EP, NW, NC, UVA, SG
        - 2-Letter Series (WP GA-1234) & 3-Letter Series (WP CAB-1234)
        - Context-aware character replacement (0 vs O, 8 vs B)
        """
        # 1. Clean and Normalize
        # Remove common delimiters and spaces to get raw sequence
        raw = text.upper().replace("-", "").replace(" ", "").replace(".", "")
        clean = re.sub(r'[^A-Z0-9]', '', raw)
        
        # If too short, probably not a valid plate query
        if len(clean) < 5:
            return text

        # 2. Identify Province
        # Provinces: WP, CP, SP, NP, EP, NW, NC, SG (2 chars) + UVA (3 chars)
        provinces_2 = ["WP", "CP", "SP", "NP", "EP", "NW", "NC", "SG"]
        provinces_3 = ["UVA"]
        
        detected_province = ""
        body = clean

        # Check for 3-letter province first
        if len(clean) >= 7 and clean[:3] in provinces_3:
            detected_province = clean[:3]
            body = clean[3:]
        # Check for 2-letter province
        elif len(clean) >= 6 and clean[:2] in provinces_2:
            detected_province = clean[:2]
            body = clean[2:]
        else:
            # Fallback: Try to fix likely OCR errors in province (e.g., VP -> WP)
            # This is risky without strict confidence, but let's try a few common ones
            prefix2 = clean[:2]
            if prefix2.replace('V', 'W') in provinces_2: # VP -> WP
                detected_province = prefix2.replace('V', 'W')
                body = clean[2:]
            elif len(clean) >= 3 and clean[:3] == "UVA": # Just in case
                 detected_province = "UVA"
                 body = clean[3:]

        # 3. Parse Body (Letters + Numbers)
        # Expected format: letters (2-3) + numbers (4)
        # We assume the LAST 4 characters are numbers.
        
        if len(body) < 4:
            return text # Structure is broken

        numbers_part = body[-4:]
        letters_part = body[:-4]
        
        # 4. Apply Corrections
        
        # Fix Numbers: 0, 1, 8, 5, 2 etc.
        # Map letters that look like numbers to numbers
        num_map = str.maketrans({
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1', 'T': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            'A': '4',
            'G': '6' 
        })
        numbers_part = numbers_part.translate(num_map)

        # Fix Letters: O, I, Z, S, B etc.
        # Map numbers that look like letters to letters
        let_map = str.maketrans({
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '5': 'S',
            '8': 'B',
            '4': 'A',
            '6': 'G'
        })
        letters_part = letters_part.translate(let_map)
        
        # 5. Format Output
        # [PROV] [LETTERS]-[NUMBERS]
        final_plate = ""
        if detected_province:
            final_plate += f"{detected_province} "
        
        if letters_part:
            final_plate += f"{letters_part}-{numbers_part}"
        else:
            final_plate += f"{numbers_part}"
            
        return final_plate.strip()

# Test Cases
service = MockService()
tests = [
    ("WPCA1234", "WP CA-1234"),
    ("WPCAB1234", "WP CAB-1234"),
    ("WP CA 1234", "WP CA-1234"),
    ("WP-CA-1234", "WP CA-1234"),
    ("WPCA123A", "WP CA-1234"), # A -> 4 in numbers
    ("WPCABIZ34", "WP CAB-1234"), # I -> 1, Z -> 2 in numbers
    ("CA 1234", "CA-1234"), # No province
    ("UVACB1234", "UVA CB-1234"), # 3-letter province
    ("VP CA 1234", "WP CA-1234"), # VP -> WP correction
    ("WP 8AB 1234", "WP BAB-1234"), # 8 -> B in letters
]

failed = False
for inp, expected in tests:
    res = service.correct_sri_lankan_text(inp)
    if res != expected:
        print(f"❌ Failed: {inp} -> Got '{res}', Expected '{expected}'")
        failed = True
    else:
        print(f"✅ Passed: {inp} -> {res}")

if not failed:
    print("ALL TESTS PASSED")
else:
    exit(1)
