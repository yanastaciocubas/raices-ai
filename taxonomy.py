"""
taxonomy.py  Raíces cultural motif taxonomy (v1, Latin American heritage)

Each Motif entry defines:
  - id:           slug used as a stable key across the codebase
  - name:         canonical display name (English)
  - native_term:  term in the tradition's primary language
  - tradition:    broad cultural tradition (used for macro-F1 grouping)
  - category:     one of {object, setting, food, clothing, activity}
  - description:  one-sentence grounding description for human annotators
  - prompts:      2–3 CLIP-ready text prompts for zero-shot detection
                  (variant 0 = English description,
                   variant 1 = native-language term + short gloss,
                   variant 2 = paraphrase / scene-level framing)
  - tags:         free-form list for cross-cutting analysis

Usage
-----
    from taxonomy import MOTIFS, by_tradition, by_category

    # All motifs for zero-shot evaluation
    all_prompts = [(m.id, p) for m in MOTIFS for p in m.prompts]

    # Subset by tradition
    mexican = by_tradition("Mexican")
"""

from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────

@dataclass
class Motif:
    id: str
    name: str
    native_term: str
    tradition: str
    category: str                        # object | setting | food | clothing | activity
    description: str
    prompts: List[str]
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None          # annotation guidance for labelers

    def __post_init__(self):
        assert self.category in {"object", "setting", "food", "clothing", "activity"}, \
            f"Invalid category '{self.category}' for motif '{self.id}'"
        assert 2 <= len(self.prompts) <= 3, \
            f"Motif '{self.id}' must have 2–3 prompts, got {len(self.prompts)}"
        assert self.id == self.id.lower().replace(" ", "_"), \
            f"ID '{self.id}' must be snake_case"


# ──────────────────────────────────────────────
# Taxonomy (v1, Latin American heritage, ~70 motifs)
# ──────────────────────────────────────────────

MOTIFS: List[Motif] = [

    # ── Mexican ──────────────────────────────────────────────────────────

    Motif(
        id="papel_picado",
        name="Papel Picado",
        native_term="papel picado",
        tradition="Mexican",
        category="object",
        description="Colorful perforated tissue-paper banners hung decoratively, associated with Día de los Muertos and fiestas.",
        prompts=[
            "colorful perforated paper banners hanging as decorations",
            "papel picado, decorative cut paper flags strung across a space",
            "intricate tissue paper cutout decorations for Mexican celebrations",
        ],
        tags=["dia_de_los_muertos", "festive", "handcraft"],
    ),

    Motif(
        id="ofrenda",
        name="Ofrenda",
        native_term="ofrenda",
        tradition="Mexican",
        category="setting",
        description="A tiered altar built during Día de los Muertos, bearing photographs of the deceased, marigolds, food, and candles.",
        prompts=[
            "a Día de los Muertos altar with photos, flowers, and candles",
            "ofrenda, a tiered memorial altar with marigolds and offerings",
            "decorated home altar honoring deceased ancestors with food and photographs",
        ],
        tags=["dia_de_los_muertos", "ritual", "altar"],
    ),

    Motif(
        id="cempasuchil",
        name="Cempasúchil",
        native_term="cempasúchil",
        tradition="Mexican",
        category="object",
        description="Mexican marigold (Tagetes erecta), used to create petal paths and decorate ofrendas during Día de los Muertos.",
        prompts=[
            "bright orange Mexican marigold flowers used in Day of the Dead",
            "cempasúchil, vivid orange and yellow marigolds forming a petal path",
            "marigold petals arranged as a trail leading to a memorial altar",
        ],
        tags=["dia_de_los_muertos", "flower", "ritual"],
    ),

    Motif(
        id="calavera_sugar",
        name="Sugar Skull",
        native_term="calavera de azúcar",
        tradition="Mexican",
        category="object",
        description="Decorated sugar skull ornament associated with Día de los Muertos, often hand-painted and inscribed with a name.",
        prompts=[
            "a decorated sugar skull with colorful painted designs",
            "calavera de azúcar, an ornamental skull decorated for Day of the Dead",
            "hand-painted candy skull with floral patterns and bright icing",
        ],
        tags=["dia_de_los_muertos", "craft", "symbol"],
    ),

    Motif(
        id="catrina",
        name="La Catrina",
        native_term="La Catrina",
        tradition="Mexican",
        category="object",
        description="Iconic skeleton figure dressed in elegant early-20th-century attire, symbol of Día de los Muertos.",
        prompts=[
            "La Catrina, an elegant skeleton figure in a wide-brimmed hat",
            "a dressed-up skeleton wearing Victorian-era clothing for Day of the Dead",
            "ornate skull figure in a feathered hat, a Mexican cultural icon",
        ],
        tags=["dia_de_los_muertos", "symbol", "art"],
    ),

    Motif(
        id="huipil",
        name="Huipil",
        native_term="huipil",
        tradition="Mexican",
        category="clothing",
        description="Traditional Mesoamerican tunic-style garment worn by Indigenous women, decorated with woven or embroidered motifs.",
        prompts=[
            "a huipil, a traditional embroidered tunic worn by Indigenous Mexican women",
            "colorfully embroidered rectangular tunic garment from Mesoamerica",
            "woman wearing a traditional woven blouse with geometric Indigenous patterns",
        ],
        tags=["indigenous", "textile", "clothing"],
    ),

    Motif(
        id="rebozo",
        name="Rebozo",
        native_term="rebozo",
        tradition="Mexican",
        category="clothing",
        description="Long woven shawl or scarf, a traditional Mexican garment used for warmth, ceremony, and carrying.",
        prompts=[
            "a woman wearing a traditional Mexican woven shawl called a rebozo",
            "rebozo, a long handwoven wrap draped over the shoulders",
            "striped or patterned Mexican cloth wrap used as a shawl or baby carrier",
        ],
        tags=["textile", "clothing"],
    ),

    Motif(
        id="talavera",
        name="Talavera Pottery",
        native_term="talavera",
        tradition="Mexican",
        category="object",
        description="Distinctive hand-painted blue-and-white (and multicolor) earthenware from Puebla, Mexico.",
        prompts=[
            "Talavera pottery with hand-painted blue and white floral designs",
            "talavera, colorful Mexican earthenware with intricate painted patterns",
            "a ceramic tile or pot with traditional Puebla-style blue and white decoration",
        ],
        tags=["pottery", "craft", "Puebla"],
    ),

    Motif(
        id="alebrijes",
        name="Alebrijes",
        native_term="alebrijes",
        tradition="Mexican",
        category="object",
        description="Brightly painted fantastical animal sculptures from Oaxaca, combining features of multiple creatures.",
        prompts=[
            "alebrijes, brightly painted fantastical Mexican folk art animal sculptures",
            "a colorful carved wooden creature with intricate painted patterns from Oaxaca",
            "whimsical hand-painted hybrid animal figurines in vivid colors",
        ],
        tags=["folk_art", "Oaxaca", "craft"],
    ),

    Motif(
        id="lucha_libre_mask",
        name="Lucha Libre Mask",
        native_term="máscara de lucha libre",
        tradition="Mexican",
        category="object",
        description="Colorful wrestling mask worn by Mexican lucha libre performers, an icon of popular culture.",
        prompts=[
            "a colorful Mexican lucha libre wrestling mask",
            "máscara de lucha libre, a bright fabric mask worn by Mexican wrestlers",
            "a vividly decorated mask associated with Mexican wrestling culture",
        ],
        tags=["sport", "popular_culture", "mask"],
    ),

    Motif(
        id="mariachi_outfit",
        name="Mariachi Outfit",
        native_term="traje de charro",
        tradition="Mexican",
        category="clothing",
        description="Ornate charro suit worn by mariachi musicians, typically black or dark with silver embroidery.",
        prompts=[
            "a mariachi musician wearing a traditional charro suit with silver trim",
            "traje de charro, an embroidered black suit worn in Mexican mariachi music",
            "ornate wide-brimmed sombrero and embroidered jacket of a mariachi band member",
        ],
        tags=["music", "clothing", "charro"],
    ),

    Motif(
        id="tortilla_making",
        name="Tortilla Making",
        native_term="tortear / hacer tortillas",
        tradition="Mexican",
        category="activity",
        description="The traditional practice of hand-pressing or patting corn masa into tortillas on a comal.",
        prompts=[
            "hands pressing corn dough into a tortilla on a traditional comal",
            "tortear, a person hand-making corn tortillas on a flat griddle",
            "making fresh tortillas by patting masa by hand over a fire or stove",
        ],
        tags=["food", "corn", "ritual", "everyday"],
    ),

    Motif(
        id="mole",
        name="Mole",
        native_term="mole",
        tradition="Mexican",
        category="food",
        description="Rich Mexican sauce made from dried chiles, chocolate, and spices, typically served over meat with rice.",
        prompts=[
            "a plate of mole sauce, a dark rich Mexican chile-and-chocolate sauce",
            "mole negro served over chicken with rice, a traditional Oaxacan dish",
            "dark brown mole sauce coating turkey or chicken on a ceramic plate",
        ],
        tags=["food", "Oaxaca", "Puebla"],
    ),

    Motif(
        id="tamales",
        name="Tamales",
        native_term="tamales",
        tradition="Mexican",
        category="food",
        description="Steamed masa parcels wrapped in corn husks or banana leaves, filled with meat, cheese, or sweet fillings.",
        prompts=[
            "tamales wrapped in corn husks steaming in a pot",
            "tamales, masa dough parcels in corn husk wrappers ready to eat",
            "unwrapped tamale showing corn dough and filling on a plate",
        ],
        tags=["food", "corn", "celebration"],
    ),

    Motif(
        id="pozole",
        name="Pozole",
        native_term="pozole",
        tradition="Mexican",
        category="food",
        description="Traditional Mexican soup made with hominy corn and meat, topped with shredded cabbage, radish, and lime.",
        prompts=[
            "a bowl of pozole, Mexican hominy soup with shredded pork and garnishes",
            "pozole rojo, a red broth soup with large corn kernels and cabbage topping",
            "traditional Mexican soup with hominy, lime wedge, radishes, and oregano",
        ],
        tags=["food", "corn", "festive"],
    ),

    Motif(
        id="cenote",
        name="Cenote",
        native_term="cenote",
        tradition="Mexican",
        category="setting",
        description="Natural sinkhole filled with clear water, sacred to the ancient Maya and found throughout the Yucatán Peninsula.",
        prompts=[
            "a cenote, a natural limestone sinkhole filled with crystal-clear water in the Yucatán",
            "cenote swimming hole with turquoise water, roots, and stalactites overhead",
            "an underground cenote pool with light filtering through an opening above",
        ],
        tags=["Maya", "nature", "sacred", "Yucatan"],
    ),

    Motif(
        id="molcajete",
        name="Molcajete",
        native_term="molcajete",
        tradition="Mexican",
        category="object",
        description="Stone mortar and pestle made from volcanic rock, used for grinding spices and making salsas.",
        prompts=[
            "a molcajete, a volcanic stone mortar and pestle used for grinding salsa",
            "molcajete filled with freshly ground guacamole or salsa",
            "traditional Mexican stone grinding bowl with pestle on a kitchen surface",
        ],
        tags=["food", "tool", "kitchen"],
    ),

    # ── Guatemalan / Maya ─────────────────────────────────────────────────

    Motif(
        id="traje_indigena_guatemalteco",
        name="Guatemalan Indigenous Traje",
        native_term="traje indígena",
        tradition="Guatemalan",
        category="clothing",
        description="Handwoven traditional garments worn by Maya communities in Guatemala, with intricate patterns denoting community of origin.",
        prompts=[
            "a Guatemalan woman wearing traditional handwoven Maya traje with colorful patterns",
            "traje indígena, a traditional Guatemalan woven blouse and skirt with geometric designs",
            "huipil and corte ensemble worn by a Maya woman from Guatemala with symbolic woven motifs",
        ],
        tags=["Maya", "indigenous", "textile", "clothing"],
    ),

    Motif(
        id="quetzal",
        name="Quetzal Bird",
        native_term="quetzal",
        tradition="Guatemalan",
        category="object",
        description="Resplendent quetzal (Pharomachrus mocinno), a sacred bird of the Maya, with brilliant green and red plumage and long tail feathers.",
        prompts=[
            "a resplendent quetzal bird with bright green and red feathers and long tail",
            "quetzal, the sacred Mesoamerican bird perched in a cloud forest",
            "a vividly colored green-and-red quetzal with long iridescent tail plumes",
        ],
        tags=["Maya", "symbol", "bird", "nature"],
    ),

    Motif(
        id="chichicastenango_market",
        name="Chichicastenango Market",
        native_term="mercado de Chichicastenango",
        tradition="Guatemalan",
        category="setting",
        description="The famous open-air indigenous market in Chichicastenango, Guatemala, renowned for handwoven textiles and ceremonial goods.",
        prompts=[
            "the open-air market of Chichicastenango with colorful textiles and vendors",
            "mercado indígena in Guatemala with woven fabrics and traditional crafts on display",
            "a crowded Guatemalan market with rows of colorful woven goods and incense smoke",
        ],
        tags=["Maya", "market", "textile", "setting"],
    ),

    Motif(
        id="jade_maya",
        name="Maya Jade Artifact",
        native_term="jade maya",
        tradition="Guatemalan",
        category="object",
        description="Carved jade ornaments or jewelry sacred to the ancient Maya, including masks, beads, and pectorals.",
        prompts=[
            "a carved jade Maya ornament or jade bead necklace",
            "jade maya, a green stone artifact carved with Mesoamerican motifs",
            "ancient Maya jade mask or jade pectoral jewelry piece",
        ],
        tags=["Maya", "artifact", "sacred", "jewelry"],
    ),

    # ── Peruvian ──────────────────────────────────────────────────────────

    Motif(
        id="poncho_andino",
        name="Andean Poncho",
        native_term="poncho andino",
        tradition="Peruvian",
        category="clothing",
        description="Woven wool outer garment with a central neck opening, worn across the Andean highlands.",
        prompts=[
            "a person wearing a traditional Andean wool poncho with geometric woven patterns",
            "poncho andino, a brightly striped Peruvian woven garment draped over the shoulders",
            "traditional Quechua poncho with colorful stripes and diamond motifs",
        ],
        tags=["Andean", "textile", "Quechua", "clothing"],
    ),

    Motif(
        id="chullo",
        name="Chullo",
        native_term="chullo",
        tradition="Peruvian",
        category="clothing",
        description="Traditional Andean knitted hat with ear flaps, decorated with colorful Andean patterns.",
        prompts=[
            "a chullo, a traditional Andean knitted hat with ear flaps and colorful patterns",
            "a person wearing a brightly decorated woolen Peruvian hat with earflaps",
            "chullo andino, a hand-knit hat with geometric designs and ear flap ties",
        ],
        tags=["Andean", "textile", "Quechua", "clothing"],
    ),

    Motif(
        id="machu_picchu",
        name="Machu Picchu",
        native_term="Machu Picchu",
        tradition="Peruvian",
        category="setting",
        description="Inca citadel set high in the Andes Mountains above Urubamba Valley, a UNESCO World Heritage Site.",
        prompts=[
            "the Inca ruins of Machu Picchu perched on an Andean mountain ridge",
            "Machu Picchu, the ancient stone citadel surrounded by mist and Andean peaks",
            "terraced Inca ruins on a cloud-covered mountain with green slopes below",
        ],
        tags=["Inca", "heritage_site", "architecture", "Andean"],
    ),

    Motif(
        id="llama",
        name="Llama",
        native_term="llama",
        tradition="Peruvian",
        category="object",
        description="Domesticated South American camelid, integral to Andean culture for transport, wool, and ceremony.",
        prompts=[
            "a llama standing in an Andean landscape or highland setting",
            "llama, a domesticated South American camelid with a decorated neck and Andean mountains behind",
            "a llama or alpaca with traditional Andean woven blanket and ear tassels",
        ],
        tags=["Andean", "animal", "symbol"],
    ),

    Motif(
        id="alpaca_wool",
        name="Alpaca Wool Weaving",
        native_term="tejido de alpaca",
        tradition="Peruvian",
        category="object",
        description="Handwoven or hand-knit textiles made from alpaca fiber, known for warmth and intricate patterning.",
        prompts=[
            "handwoven Andean textiles made from alpaca wool with colorful geometric patterns",
            "tejido de alpaca, a traditional Peruvian woven cloth with bold zigzag motifs",
            "a market stall or weaver displaying alpaca fiber blankets, scarves, or ponchos",
        ],
        tags=["Andean", "textile", "craft"],
    ),

    Motif(
        id="ceviche",
        name="Ceviche",
        native_term="ceviche",
        tradition="Peruvian",
        category="food",
        description="Raw seafood cured in citrus juice, mixed with onion, chili, and cilantro. Peru's national dish.",
        prompts=[
            "a plate of Peruvian ceviche with raw fish marinated in lime, red onion, and ají amarillo",
            "ceviche peruano served in a bowl with corn, sweet potato, and cilantro",
            "fresh seafood ceviche with chili and sliced red onion in a citrus marinade",
        ],
        tags=["food", "seafood", "national_dish"],
    ),

    Motif(
        id="inca_sun_symbol",
        name="Inca Sun Symbol (Inti)",
        native_term="Inti",
        tradition="Peruvian",
        category="object",
        description="Radiant sun deity emblem of the Inca, depicted as a face with radiating rays, found on textiles, metalwork, and the Peruvian flag.",
        prompts=[
            "the Inca sun symbol Inti, a golden face with radiating rays",
            "Inti, the Inca sun god depicted as a radiant circle with a human face",
            "a golden sunburst emblem from Inca culture on a woven textile or metal object",
        ],
        tags=["Inca", "symbol", "sacred", "metalwork"],
    ),

    Motif(
        id="quipu",
        name="Quipu",
        native_term="quipu",
        tradition="Peruvian",
        category="object",
        description="Inca recording device consisting of knotted colored cords used to encode information.",
        prompts=[
            "a quipu, an Inca knotted cord device used for recording data",
            "quipu, hanging colored strings with knots used by the Inca as a recording system",
            "an ancient Andean knotted textile information system made of dyed wool cords",
        ],
        tags=["Inca", "artifact", "knowledge"],
    ),

    Motif(
        id="caballito_de_totora",
        name="Caballito de Totora",
        native_term="caballito de totora",
        tradition="Peruvian",
        category="object",
        description="Traditional reed fishing boat used by fishermen on the northern Peruvian coast near Huanchaco.",
        prompts=[
            "a caballito de totora, a traditional reed boat used by Peruvian coastal fishermen",
            "totora reed watercraft ridden like a surfboard by a fisherman at Huanchaco beach",
            "a bundled reed fishing vessel from northern Peru's pre-Hispanic coastal tradition",
        ],
        tags=["coastal", "fishing", "craft", "pre-Hispanic"],
    ),

    # ── Colombian ─────────────────────────────────────────────────────────

    Motif(
        id="mochila_wayuu",
        name="Wayuu Mochila Bag",
        native_term="mochila wayuú",
        tradition="Colombian",
        category="object",
        description="Hand-crocheted bag made by Wayuu women of the Guajira Peninsula, with intricate geometric patterns.",
        prompts=[
            "a Wayuu mochila bag, a hand-crocheted Colombian bag with bold geometric patterns",
            "mochila wayuú, a brightly colored crochet shoulder bag from the Guajira region",
            "a traditional Colombian woven bag with multicolor diamond and zigzag patterns",
        ],
        tags=["Wayuu", "indigenous", "textile", "craft"],
    ),

    Motif(
        id="sombrero_vueltiao",
        name="Sombrero Vueltiao",
        native_term="sombrero vueltiao",
        tradition="Colombian",
        category="clothing",
        description="Colombian national hat made from caña flecha fiber, woven with black-and-white geometric patterns.",
        prompts=[
            "a sombrero vueltiao, a traditional Colombian woven hat with black-and-white geometric design",
            "the iconic Colombian national hat made from woven cane fiber",
            "a black-and-white striped brimmed hat from Colombia's Atlantic coast region",
        ],
        tags=["clothing", "symbol", "Atlantic_coast"],
    ),

    Motif(
        id="cumbia_dance",
        name="Cumbia Dance",
        native_term="cumbia",
        tradition="Colombian",
        category="activity",
        description="Traditional Colombian folk dance featuring women in flowing skirts holding candles and men in straw hats.",
        prompts=[
            "dancers performing cumbia, a traditional Colombian folk dance with flowing skirts",
            "cumbia, Colombian cultural dance where women wave colorful skirts and hold candles",
            "a couple dancing cumbia in festive traditional Colombian clothing",
        ],
        tags=["music", "dance", "folk"],
    ),

    Motif(
        id="arepas",
        name="Arepas",
        native_term="arepas",
        tradition="Colombian",
        category="food",
        description="Thick corn patties grilled or fried, a staple food of Colombia and Venezuela.",
        prompts=[
            "arepas, round grilled corn patties on a griddle or plate",
            "a Colombian arepa stuffed with cheese or meat on a wooden board",
            "freshly made corn flour flatbreads cooking on a comal or griddle",
        ],
        tags=["food", "corn", "staple"],
    ),

    Motif(
        id="ciudad_perdida",
        name="Ciudad Perdida",
        native_term="Ciudad Perdida",
        tradition="Colombian",
        category="setting",
        description="Pre-Columbian archaeological site of the Tayrona civilization in the Sierra Nevada de Santa Marta.",
        prompts=[
            "the terraced stone ruins of Ciudad Perdida in the Colombian jungle",
            "Ciudad Perdida, an ancient Tayrona archaeological site with circular stone platforms and stone steps",
            "moss-covered stone terraces emerging from dense jungle in Colombia's Sierra Nevada",
        ],
        tags=["Tayrona", "archaeology", "heritage_site"],
    ),

    # ── Cuban ─────────────────────────────────────────────────────────────

    Motif(
        id="son_cubano",
        name="Son Cubano Performance",
        native_term="son cubano",
        tradition="Cuban",
        category="activity",
        description="Traditional Cuban musical genre combining Spanish and African influences, performed with guitar, tres, and percussion.",
        prompts=[
            "musicians playing son cubano with guitars, percussion, and bass in a Cuban setting",
            "a traditional Cuban son ensemble performing on a colorful Havana street",
            "son cubano music performance with claves, guitar, and singers in Cuba",
        ],
        tags=["music", "Afro-Cuban", "performance"],
    ),

    Motif(
        id="habana_vieja_architecture",
        name="Old Havana Architecture",
        native_term="arquitectura de La Habana Vieja",
        tradition="Cuban",
        category="setting",
        description="Colonial Spanish architecture of Old Havana (La Habana Vieja), characterized by arcaded facades, pastel colors, and wrought iron balconies.",
        prompts=[
            "colonial pastel-colored buildings with wrought iron balconies in Old Havana",
            "La Habana Vieja, a street of Spanish colonial architecture with arched walkways in Cuba",
            "faded but colorful Havana streetscape with vintage American cars and crumbling facades",
        ],
        tags=["architecture", "colonial", "UNESCO", "Havana"],
    ),

    Motif(
        id="guayabera",
        name="Guayabera Shirt",
        native_term="guayabera",
        tradition="Cuban",
        category="clothing",
        description="Traditional men's shirt with vertical pleats and embroidery, worn throughout Cuba and Latin America.",
        prompts=[
            "a man wearing a guayabera, a traditional Cuban pleated linen shirt",
            "guayabera, a formal lightweight shirt with vertical tuck rows and embroidery",
            "a white or light-colored guayabera shirt associated with Cuban and Caribbean formal dress",
        ],
        tags=["clothing", "Caribbean"],
    ),

    Motif(
        id="mojito",
        name="Mojito",
        native_term="mojito",
        tradition="Cuban",
        category="food",
        description="Classic Cuban cocktail made with rum, lime, mint, sugar, and soda water.",
        prompts=[
            "a mojito cocktail with fresh mint, lime, and ice in a tall glass",
            "mojito cubano, a rum-based drink garnished with lime and mint sprigs",
            "a classic Cuban highball with muddled mint and lime slices in a glass",
        ],
        tags=["food", "drink", "cocktail"],
    ),

    # ── Argentine ────────────────────────────────────────────────────────

    Motif(
        id="mate_gourd",
        name="Mate Gourd",
        native_term="mate",
        tradition="Argentine",
        category="object",
        description="Gourd vessel used for drinking yerba mate through a metal straw (bombilla), a central ritual of Argentine and Uruguayan daily life.",
        prompts=[
            "a mate gourd with a metal bombilla straw filled with yerba mate leaves",
            "mate, a traditional South American gourd cup used to drink herbal infusion",
            "someone holding a carved calabash gourd of mate with a silver bombilla",
        ],
        tags=["drink", "ritual", "daily_life", "Río_de_la_Plata"],
    ),

    Motif(
        id="tango_dance",
        name="Tango",
        native_term="tango",
        tradition="Argentine",
        category="activity",
        description="Intimate Argentine partnered dance originating in Buenos Aires, characterized by close embrace and intricate footwork.",
        prompts=[
            "a couple dancing tango in close embrace on a Buenos Aires street or milonga",
            "tango, an Argentine partnered dance with dramatic poses and intertwined leg movements",
            "tango dancers in formal attire performing in a dimly lit Porteño dance hall",
        ],
        tags=["dance", "music", "Buenos_Aires", "UNESCO"],
    ),

    Motif(
        id="asado",
        name="Asado",
        native_term="asado",
        tradition="Argentine",
        category="activity",
        description="Traditional Argentine barbecue gathering where meat is slow-cooked over wood or charcoal on a parrilla grill.",
        prompts=[
            "an asado, a traditional Argentine barbecue with cuts of meat grilling on a parrilla",
            "asado argentino, beef ribs and sausages cooking over glowing coals outdoors",
            "a parrilla grill loaded with chimichurri-seasoned meat at an Argentine outdoor gathering",
        ],
        tags=["food", "ritual", "social", "beef"],
    ),

    Motif(
        id="gaucho_attire",
        name="Gaucho Attire",
        native_term="vestimenta gaucha",
        tradition="Argentine",
        category="clothing",
        description="Traditional clothing of the gaucho, the South American cowboy: baggy bombacha trousers, wide belt, boots, and beret.",
        prompts=[
            "a gaucho wearing traditional Argentine bombachas, a wide leather belt, and boots",
            "vestimenta gaucha, a South American cowboy outfit with wide trousers and a beret",
            "a traditional Argentine gaucho on horseback wearing a poncho and wide-brimmed hat",
        ],
        tags=["gaucho", "clothing", "Pampas"],
    ),

    Motif(
        id="empanadas",
        name="Empanadas",
        native_term="empanadas",
        tradition="Argentine",
        category="food",
        description="Baked or fried stuffed pastry pockets filled with seasoned beef, chicken, or cheese, common across Latin America.",
        prompts=[
            "Argentine empanadas, baked or fried filled pastry pockets on a plate",
            "empanadas argentinas with crimped edges on a wooden board, one cut open to show filling",
            "golden pastry half-moons stuffed with spiced beef or chicken",
        ],
        tags=["food", "pastry", "widespread"],
    ),

    Motif(
        id="yerba_mate_ritual",
        name="Yerba Mate Sharing Ritual",
        native_term="ronda de mate",
        tradition="Argentine",
        category="activity",
        description="The social ritual of passing a mate gourd in a circle among friends or family, refilling with hot water from a thermos.",
        prompts=[
            "a group of people sharing mate in a circle, passing the gourd with a thermos nearby",
            "ronda de mate, the Argentine social ritual of passing a mate cup among friends",
            "people seated outdoors passing a mate gourd and thermos of hot water",
        ],
        tags=["ritual", "social", "drink", "Río_de_la_Plata"],
    ),

    # ── Brazilian ─────────────────────────────────────────────────────────

    Motif(
        id="capoeira",
        name="Capoeira",
        native_term="capoeira",
        tradition="Brazilian",
        category="activity",
        description="Afro-Brazilian martial art combining dance, acrobatics, and music, performed to the sound of the berimbau.",
        prompts=[
            "capoeira practitioners performing acrobatic martial arts moves in a roda circle",
            "capoeira, an Afro-Brazilian dance-fight art with two players and a berimbau musician",
            "two capoeiristas in white uniforms doing a ginga and kick sequence outdoors",
        ],
        tags=["Afro-Brazilian", "martial_art", "dance", "music"],
    ),

    Motif(
        id="carnival_costume",
        name="Brazilian Carnival Costume",
        native_term="fantasia de carnaval",
        tradition="Brazilian",
        category="clothing",
        description="Elaborate feathered and sequined costume worn by samba school performers during Brazilian Carnival.",
        prompts=[
            "an elaborate Brazilian Carnival costume with feathers, sequins, and headdress",
            "fantasia de carnaval, a samba dancer in a glittering feathered Carnival outfit",
            "a performer at the Rio Carnival sambadrome in a massive colorful feathered costume",
        ],
        tags=["carnival", "samba", "Rio", "clothing"],
    ),

    Motif(
        id="berimbau",
        name="Berimbau",
        native_term="berimbau",
        tradition="Brazilian",
        category="object",
        description="Single-string musical bow instrument central to capoeira music, played with a stone or coin and a rattle.",
        prompts=[
            "a berimbau, a single-string musical bow played in Brazilian capoeira",
            "berimbau instrument with a gourd resonator being played with a stone and stick",
            "a musician playing a berimbau alongside a pandeiro drum in a capoeira roda",
        ],
        tags=["Afro-Brazilian", "music", "instrument", "capoeira"],
    ),

    Motif(
        id="feijoada",
        name="Feijoada",
        native_term="feijoada",
        tradition="Brazilian",
        category="food",
        description="Brazil's national dish: a hearty black bean stew slow-cooked with various cuts of pork, served with rice and farofa.",
        prompts=[
            "feijoada, a Brazilian black bean and pork stew served with rice and orange slices",
            "a clay pot of feijoada with farofa, collard greens, and rice on the side",
            "Brazil's national dish of slow-cooked black beans with smoked pork ribs and sausage",
        ],
        tags=["food", "Afro-Brazilian", "national_dish"],
    ),

    Motif(
        id="candomble_ceremony",
        name="Candomblé Ceremony",
        native_term="candomblé",
        tradition="Brazilian",
        category="activity",
        description="Afro-Brazilian religious ceremony honoring orixás (spirits), featuring elaborate dress, drumming, and ritual dance.",
        prompts=[
            "a candomblé ceremony with participants in white clothing and ritual drumming",
            "candomblé, an Afro-Brazilian spiritual gathering with colorful beaded necklaces and offerings",
            "a terreiro de candomblé ritual with women in ceremonial white and flower offerings",
        ],
        tags=["Afro-Brazilian", "religion", "ritual"],
    ),

    Motif(
        id="favela_mural",
        name="Favela Mural / Street Art",
        native_term="mural de favela",
        tradition="Brazilian",
        category="setting",
        description="Vibrant street murals painted on the densely packed hillside houses of Rio de Janeiro favelas.",
        prompts=[
            "colorful murals painted on the stacked houses of a Rio de Janeiro favela",
            "mural de favela, large street art covering the exterior walls of a Brazilian hillside community",
            "a dense cluster of painted concrete homes in a Rio favela with a large mural",
        ],
        tags=["urban", "street_art", "Rio", "community"],
    ),

    # ── Chilean ──────────────────────────────────────────────────────────

    Motif(
        id="mapuche_kultrun",
        name="Mapuche Kultrun",
        native_term="kultrun",
        tradition="Chilean",
        category="object",
        description="Sacred ceremonial drum of the Mapuche people, decorated with cosmic symbols and used by the machi (spiritual healer).",
        prompts=[
            "a kultrun, the sacred Mapuche ceremonial drum with painted cosmic symbols",
            "kultrun mapuche, a painted drum used in Mapuche spiritual healing ceremonies",
            "a round hand drum with symbolic cross and circle motifs from southern Chile",
        ],
        tags=["Mapuche", "indigenous", "instrument", "sacred"],
    ),

    Motif(
        id="cueca_dance",
        name="Cueca Dance",
        native_term="cueca",
        tradition="Chilean",
        category="activity",
        description="Chile's national dance, a handkerchief-waving courtship dance performed in huaso (horseman) attire.",
        prompts=[
            "a couple dancing cueca, Chile's national folk dance with white handkerchiefs",
            "cueca chilena, a traditional courtship dance where dancers wave handkerchiefs",
            "dancers in huaso and china attire performing cueca at a Chilean fonda",
        ],
        tags=["dance", "folk", "national_dance"],
    ),

    # ── Bolivian ─────────────────────────────────────────────────────────

    Motif(
        id="cholita_dress",
        name="Cholita Dress",
        native_term="vestimenta de cholita",
        tradition="Bolivian",
        category="clothing",
        description="Traditional clothing of Aymara and Quechua women in Bolivia: full layered skirts (pollera), shawl, and bowler hat.",
        prompts=[
            "a Bolivian cholita wearing a pollera skirt, shawl, and bowler hat",
            "vestimenta de cholita, a Bolivian indigenous woman in layered colorful skirts and a derby hat",
            "an Aymara woman in La Paz wearing traditional pollera, aguayo shawl, and black bowler hat",
        ],
        tags=["Aymara", "Quechua", "clothing", "Bolivia"],
    ),

    Motif(
        id="tinku_ritual",
        name="Tinku Ritual",
        native_term="tinku",
        tradition="Bolivian",
        category="activity",
        description="Andean ritual encounter involving ceremonial combat and colorful costumes, held in the northern Potosí region.",
        prompts=[
            "tinku festival participants in colorful helmets and costumes in a Bolivian highland town",
            "tinku, a Bolivian Andean ritual festival with dancers in bright woven attire",
            "a traditional tinku ceremony in Macha, Bolivia with participants in distinctive crested helmets",
        ],
        tags=["Andean", "ritual", "festival", "Potosi"],
    ),

    Motif(
        id="salt_flat_salar",
        name="Salar de Uyuni",
        native_term="Salar de Uyuni",
        tradition="Bolivian",
        category="setting",
        description="The world's largest salt flat in southwest Bolivia, creating a mirror-like surface during the wet season.",
        prompts=[
            "the Salar de Uyuni, a vast flat salt lake in Bolivia reflecting the sky like a mirror",
            "Bolivia's salt flats with a thin layer of water creating a perfect sky reflection",
            "an endless white salt crust surface in the Bolivian Altiplano under a blue sky",
        ],
        tags=["landscape", "natural_landmark", "Bolivia"],
    ),

    # ── Puerto Rican ──────────────────────────────────────────────────────

    Motif(
        id="vejigante_mask",
        name="Vejigante Mask",
        native_term="máscara vejigante",
        tradition="Puerto Rican",
        category="object",
        description="Colorful horned papier-mâché or coconut shell mask worn during Puerto Rican carnival festivals.",
        prompts=[
            "a vejigante mask, a brightly painted horned papier-mâché mask from Puerto Rico",
            "máscara vejigante, a colorful multi-horned carnival mask from Ponce or Loíza",
            "an elaborate Puerto Rican festival mask with dozens of painted horns and vivid patterns",
        ],
        tags=["carnival", "mask", "Afro-Puerto_Rican", "craft"],
    ),

    Motif(
        id="salsa_dance_pr",
        name="Salsa Dance",
        native_term="salsa",
        tradition="Puerto Rican",
        category="activity",
        description="Vibrant partner dance rooted in Cuban son and Puerto Rican musical traditions, central to Caribbean cultural identity.",
        prompts=[
            "a couple dancing salsa with energetic footwork and spins on a dance floor",
            "salsa dancing, a lively Caribbean partner dance with syncopated rhythms",
            "two salsa dancers in a urban or club setting performing turns and rhythmic steps",
        ],
        tags=["dance", "music", "Caribbean", "Afro-Caribbean"],
    ),

    Motif(
        id="cuatro_instrument",
        name="Puerto Rican Cuatro",
        native_term="cuatro puertorriqueño",
        tradition="Puerto Rican",
        category="object",
        description="Puerto Rico's national instrument: a small five double-stringed guitar used in jíbaro folk music.",
        prompts=[
            "a cuatro puertorriqueño, a small ten-string guitar-like instrument from Puerto Rico",
            "a musician playing the cuatro, Puerto Rico's national folk instrument",
            "a carved wooden cuatro instrument used in traditional jíbaro music",
        ],
        tags=["music", "instrument", "national_symbol"],
    ),

    # ── Andean / Pan-regional ─────────────────────────────────────────────

    Motif(
        id="pan_flute_siku",
        name="Andean Pan Flute (Siku)",
        native_term="siku / zampoña",
        tradition="Andean",
        category="object",
        description="Traditional reed panpipe from the Andean highlands, played solo or in interlocking pairs.",
        prompts=[
            "a siku or zampoña, a traditional Andean pan flute made of bamboo reeds",
            "a musician playing an Andean panpipe with multiple tubes of varying lengths",
            "siku, a traditional Indigenous wind instrument from the Andean highlands",
        ],
        tags=["music", "instrument", "Andean", "Aymara"],
    ),

    Motif(
        id="aguayo_textile",
        name="Aguayo Textile",
        native_term="aguayo",
        tradition="Andean",
        category="object",
        description="Multicolored woven square cloth used throughout Bolivia, Peru, and Chile for carrying goods or as a shawl.",
        prompts=[
            "an aguayo, a traditional Andean woven cloth with geometric patterns in bright colors",
            "aguayo textile being used as a carrying cloth or shawl in the Andes",
            "a folded square of Andean woven fabric with repeating diamond and stripe motifs",
        ],
        tags=["textile", "Andean", "Aymara", "Quechua"],
    ),

    Motif(
        id="coca_leaf_ritual",
        name="Coca Leaf Ritual",
        native_term="ceremonia de la hoja de coca",
        tradition="Andean",
        category="activity",
        description="The sacred Andean practice of offering, chewing, or reading coca leaves in ceremonies, connected to Pachamama.",
        prompts=[
            "coca leaves being offered or arranged in an Andean ritual ceremony",
            "a traditional Andean healer or elder performing a ceremony with coca leaves",
            "a woven cloth spread with dried coca leaves for a Pachamama offering",
        ],
        tags=["ritual", "Andean", "sacred", "Pachamama"],
        notes="Depict only as ritual/cultural practice; images should show ceremonial context, not narcotics framing.",
    ),

    Motif(
        id="pachamama_offering",
        name="Pachamama Offering (Pago a la Tierra)",
        native_term="pago a la tierra",
        tradition="Andean",
        category="activity",
        description="Ritual offering to Pachamama (Mother Earth) involving llama fat, flowers, sweets, and tobacco, performed on August 1st.",
        prompts=[
            "a pago a la tierra ceremony with colorful ritual offerings arranged on a cloth",
            "Pachamama offering bundle with llama fat, flowers, sweets, and incense",
            "an Andean earth offering ritual with a mesa bundle and smoldering items",
        ],
        tags=["ritual", "Andean", "Pachamama", "sacred"],
    ),

]


# ──────────────────────────────────────────────
# Helper accessors
# ──────────────────────────────────────────────

def by_tradition(tradition: str) -> List[Motif]:
    """Return all motifs for a given tradition (case-insensitive)."""
    return [m for m in MOTIFS if m.tradition.lower() == tradition.lower()]


def by_category(category: str) -> List[Motif]:
    """Return all motifs for a given category."""
    return [m for m in MOTIFS if m.category == category]


def all_prompt_pairs() -> List[tuple]:
    """Return (motif_id, prompt_text) pairs for CLIP zero-shot evaluation."""
    return [(m.id, prompt) for m in MOTIFS for prompt in m.prompts]


def taxonomy_summary() -> dict:
    """Return a summary dict for quick sanity checks."""
    traditions = defaultdict(int)
    categories = defaultdict(int)
    for m in MOTIFS:
        traditions[m.tradition] += 1
        categories[m.category] += 1
    return {
        "total_motifs": len(MOTIFS),
        "total_prompts": sum(len(m.prompts) for m in MOTIFS),
        "traditions": dict(traditions),
        "categories": dict(categories),
    }


# ──────────────────────────────────────────────
# Quick self-check when run directly
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import json
    summary = taxonomy_summary()
    print(json.dumps(summary, indent=2))

    # Validate all IDs are unique
    ids = [m.id for m in MOTIFS]
    assert len(ids) == len(set(ids)), "Duplicate motif IDs detected!"
    print(f"\n✓ All {summary['total_motifs']} motif IDs are unique.")
    print(f"✓ {summary['total_prompts']} total CLIP prompt variants ready.")
