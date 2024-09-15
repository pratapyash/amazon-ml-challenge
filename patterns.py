unit_mappings = {
    # Length units
    'cm': 'centimetre', 'mm': 'millimetre', 'm': 'metre', 'in': 'inch', 'ft': 'foot', 'yd': 'yard',
    'centimetres': 'centimetre', 'millimetres': 'millimetre', 'metres': 'metre',
    'inches': 'inch', 'feet': 'foot', 'yards': 'yard',
    'cm.': 'centimetre', 'mm.': 'millimetre', 'm.': 'metre', 'in.': 'inch', 'ft.': 'foot', 'yd.': 'yard',
    '"': 'inch', "'": 'foot', 'candelas': 'candela', 'cd': 'candela', 'candela': 'candela', 'candelas': 'candela', 'candel': 'candela',

    # Weight units
    'mg': 'milligram', 'g': 'gram', 'kg': 'kilogram', 'oz': 'ounce', 'lb': 'pound', 't': 'ton',
    'μg': 'microgram', 'microg': 'microgram',
    'grams': 'gram', 'kilograms': 'kilogram', 'milligrams': 'milligram', 'ounces': 'ounce',
    'pounds': 'pound', 'tons': 'ton', 'lbs': 'pound',
    'mg.': 'milligram', 'g.': 'gram', 'kg.': 'kilogram', 'oz.': 'ounce', 'lb.': 'pound', 'lbs.': 'pound',
    'μg.': 'microgram',

    # Volume units
    'ml': 'millilitre', 'l': 'litre', 'cl': 'centilitre', 'dl': 'decilitre', 'fl oz': 'fluid ounce',
    'pt': 'pint', 'qt': 'quart', 'gal': 'gallon', 'imp gal': 'imperial gallon',
    'cu ft': 'cubic foot', 'cu in': 'cubic inch', 'cu m': 'cubic metre',
    'μl': 'microlitre', 'microl': 'microlitre', 'c': 'cup',
    'litres': 'litre', 'millilitres': 'millilitre', 'centilitres': 'centilitre', 'decilitres': 'decilitre',
    'pints': 'pint', 'quarts': 'quart', 'gallons': 'gallon', 'imperial gallons': 'imperial gallon',
    'cubic feet': 'cubic foot', 'cubic inches': 'cubic inch', 'cubic meters': 'cubic metre',
    'cubic metres': 'cubic metre', 'fluid ounces': 'fluid ounce', 'cups': 'cup',
    'ml.': 'millilitre', 'l.': 'litre', 'cl.': 'centilitre', 'dl.': 'decilitre',
    'pt.': 'pint', 'qt.': 'quart', 'gal.': 'gallon',
    'cu. ft.': 'cubic foot', 'cu. in.': 'cubic inch', 'cu. m': 'cubic metre',
    'fl. oz.': 'fluid ounce', 'fluid oz': 'fluid ounce', 'fluid oz.': 'fluid ounce',
    'us gal': 'gallon', 'us gallon': 'gallon', 'us gallons': 'gallon',
    'us fl oz': 'fluid ounce', 'us fluid ounce': 'fluid ounce', 'us fluid ounces': 'fluid ounce',
    'us pint': 'pint', 'us pints': 'pint', 'us quart': 'quart', 'us quarts': 'quart',
    'microlitres': 'microlitre',

    # Electrical units
    'mv': 'millivolt', 'kv': 'kilovolt', 'v': 'volt',
    'w': 'watt', 'kw': 'kilowatt',
    'volts': 'volt', 'millivolts': 'millivolt', 'kilovolts': 'kilovolt',
    'watts': 'watt', 'kilowatts': 'kilowatt',
    'mv.': 'millivolt', 'kv.': 'kilovolt', 'v.': 'volt',
    'w.': 'watt', 'kw.': 'kilowatt'
}

# Entity-specific patterns
entity_patterns = {
    'item_weight': r'(\d+(\.\d+)?)\s*(milligram|kilogram|microgram|gram|ounce|ton|pound|mg|kg|g|oz|lb|lbs|μg|t|microg|grams|kilograms|milligrams|ounces|pounds|tons)',
    'maximum_weight_recommendation': r'(\d+(\.\d+)?)\s*(milligram|kilogram|microgram|gram|ounce|ton|pound|mg|kg|g|oz|lb|lbs|μg|t|microg|grams|kilograms|milligrams|ounces|pounds|tons)',
    'width': r'(\d+(\.\d+)?)\s*(centimetre|foot|millimetre|metre|inch|yard|cm|mm|m|in|ft|yd|centimetres|millimetres|metres|inches|feet|yards|"|\')',
    'height': r'(\d+(\.\d+)?)\s*(centimetre|foot|millimetre|metre|inch|yard|cm|mm|m|in|ft|yd|centimetres|millimetres|metres|inches|feet|yards|"|\')',
    'depth': r'(\d+(\.\d+)?)\s*(centimetre|foot|millimetre|metre|inch|yard|cm|mm|m|in|ft|yd|centimetres|millimetres|metres|inches|feet|yards|"|\')',
    'voltage': r'(\d+(\.\d+)?)\s*(millivolt|kilovolt|volt|mv|kv|v|volts|millivolts|kilovolts)',
    'wattage': r'(\d+(\.\d+)?)\s*(kilowatt|watt|kw|w|watts|kilowatts)',
    'item_volume': r'(\d+(\.\d+)?)\s*(cubic foot|cubic metre|microlitre|cup|fluid ounce|centilitre|imperial gallon|us gallon|pint|decilitre|litre|millilitre|quart|cubic inch|gallon|cu ft|cu m|cu in|ml|l|cl|dl|fl oz|pt|qt|gal|imp gal|us gal|μl|c|microl|cubic feet|cubic inches|cubic meters|cubic metres|fluid ounces|gallons|imperial gallons|us gallons|litres|millilitres|pints|quarts|us pints|us quarts|us fluid ounces)'
}

# Contextual keyword matching
entity_keywords = {
    'item_weight': ['weight', 'weighs', 'mass', 'heavy', 'light', 'lb', 'kg', 'grams', 'ounces', 'net weight', 'gross weight', 'tare weight', 'product weight', 'item weight', 'unit weight', 'shipping weight', 'payload', 'heft', 'load', 'bulk', 'density', 'avoirdupois', 'poundage', 'tonnage', 'weightiness', 'gravitas', 'ponderosity', 'substance', 'ballast', 'burden', 'encumbrance', 'gravity', 'heaviness', 'mass', 'pressure', 'tonnage', 'weight force', 'weightage', 'dead weight', 'live weight', 'curb weight', 'dry weight', 'unladen weight', 'laden weight', 'kerb weight', 'gross vehicle weight', 'candle'],
    'maximum_weight_recommendation': ['max weight', 'maximum weight', 'weight limit', 'weight capacity', 'load capacity', 'can hold up to', 'supports up to', 'max load', 'weight rating', 'safe working load', 'recommended max weight', 'weight restriction', 'not to exceed', 'maximum load', 'weight threshold', 'upper weight limit', 'peak weight', 'weight tolerance', 'maximum carrying capacity', 'weight bearing limit', 'load limit', 'weight allowance', 'maximum permissible weight', 'weight ceiling', 'weight boundary', 'weight cutoff', 'weight maximum', 'weight cap', 'weight constraint', 'weight barrier', 'weight ceiling', 'weight threshold', 'weight upper bound', 'weight top end', 'weight peak', 'weight apex', 'weight zenith', 'weight summit', 'weight pinnacle', 'weight acme'],
    'width': ['width', 'wide', 'across', 'breadth', 'span', 'horizontal', 'side to side', 'lateral', 'diameter', 'girth', 'W:', 'W.', 'width:', 'wide:', 'cross section', 'transverse dimension', 'broadness', 'wideness', 'beam', 'thickness', 'gauge', 'caliber', 'amplitude', 'expanse', 'spread', 'broadness', 'extent', 'measurement', 'size', 'dimension', 'proportion', 'magnitude', 'scope', 'range', 'compass', 'reach', 'extension', 'expansion', 'stretch', 'span', 'latitude', 'bore', 'calibre', 'thickness', 'cross-section', 'profile'],
    'height': ['height', 'tall', 'high', 'elevation', 'vertical length', 'stature', 'altitude', 'top to bottom', 'upright', 'rise', 'H:', 'H.', 'height:', 'tall:', 'vertical dimension', 'clearance', 'tallness', 'highness', 'loftiness', 'prominence', 'eminence', 'towering', 'vertical extent', 'headroom', 'ceiling height', 'vertical distance', 'vertical measurement', 'vertical span', 'vertical reach', 'vertical dimension', 'vertical size', 'vertical proportion', 'vertical magnitude', 'vertical scope', 'vertical range', 'vertical compass', 'vertical extension', 'vertical expansion', 'vertical stretch', 'vertical elevation', 'vertical rise', 'vertical lift', 'vertical climb', 'vertical ascent', 'vertical growth'],
    'depth': ['depth', 'deep', 'thickness', 'front to back', 'length', 'extent', 'distance', 'profundity', 'dimension', 'reach', 'D:', 'D.', 'depth:', 'deep:', 'longitudinal dimension', 'deepness', 'profoundness', 'penetration', 'recession', 'inwardness', 'immersion', 'submersion', 'sinking', 'hollowness', 'concavity', 'vertical distance', 'vertical extent', 'vertical dimension', 'vertical measurement', 'vertical reach', 'vertical penetration', 'vertical recession', 'vertical immersion', 'vertical submersion', 'vertical sinking', 'vertical depression', 'vertical cavity', 'vertical hollow', 'vertical recess', 'vertical indentation', 'vertical pit', 'vertical chasm', 'vertical abyss', 'vertical gorge', 'vertical ravine'],
    'voltage': ['voltage', 'volts', 'V', 'electrical potential', 'electromotive force', 'power supply', 'input voltage', 'output voltage', 'operating voltage', 'rated voltage', 'AC voltage', 'DC voltage', 'potential difference', 'electric pressure', 'tension', 'EMF', 'volt rating', 'voltage drop', 'voltage range', 'nominal voltage', 'supply voltage', 'line voltage', 'phase voltage', 'peak voltage', 'RMS voltage', 'breakdown voltage', 'threshold voltage', 'forward voltage', 'reverse voltage', 'standoff voltage', 'surge voltage', 'ripple voltage', 'voltage regulation', 'voltage stability', 'voltage tolerance', 'voltage fluctuation', 'voltage sag', 'voltage spike', 'voltage dip', 'voltage surge', 'voltage transient'],
    'wattage': ['wattage', 'watts', 'power', 'energy consumption', 'power output', 'W', 'power rating', 'power consumption', 'energy usage', 'power draw', 'electrical power', 'rated power', 'power capacity', 'energy demand', 'power requirement', 'power level', 'energy efficiency', 'power dissipation', 'power supply', 'power specification', 'power demand', 'power input', 'power output', 'power throughput', 'power handling', 'power delivery', 'power transfer', 'power conversion', 'power generation', 'power production', 'power yield', 'power expenditure', 'power utilization', 'power allocation', 'power budget', 'power threshold', 'power limit', 'power range', 'power margin', 'power reserve'],
    'item_volume': ['volume', 'capacity', 'contains', 'content', 'holds', 'storage', 'liquid capacity', 'fluid volume', 'internal volume', 'container volume', 'total volume', 'net volume', 'gross volume', 'fill capacity', 'cubic capacity', 'volumetric capacity', 'displacement', 'interior space', 'holding capacity', 'storage space', 'cubic volume', 'spatial volume', 'volumetric content', 'volumetric measurement', 'volumetric size', 'volumetric dimension', 'volumetric extent', 'volumetric magnitude', 'volumetric quantity', 'volumetric amount', 'volumetric proportion', 'volumetric ratio', 'volumetric fraction', 'volumetric part', 'volumetric segment', 'volumetric section', 'volumetric division', 'volumetric portion', 'volumetric share', 'volumetric allotment']
}