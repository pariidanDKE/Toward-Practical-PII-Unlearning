import random
import uuid # Used for generating fake, uncommon company/bank names if lists are small

# Define the country groupings
countries_grouped = {
    "Southern & Western Europe": ["Italy", "Spain"],
    "Central & Western Europe": ["France", "Switzerland", "Germany", "Netherlands"],
    "Nordic Countries": ["Sweden", "Norway", "Denmark", "Finland"],
    "Anglophone Countries": ["United Kingdom", "US", "Canada", "Australia", "New Zealand"],
    "East Asia": ["Japan", "South Korea"],
    "Eastern Europe": ["Russia"]
}

# --- Global English Job Titles ---
# Used for all countries
global_job_titles = [
    "Engineer", "Doctor", "Lawyer", "Teacher", "Consultant", "Accountant",
    "Sales Manager", "Project Manager", "Nurse", "Architect", "Analyst",
    "Coordinator", "Specialist", "Administrator", "Scientist", "Researcher",
    "Technician", "Supervisor", "Director", "Assistant", "Manager",
    "Developer", "Designer", "Editor", "Writer", "Artist", "Musician",
    "Chef", "Baker", "Electrician", "Plumber", "Carpenter", "Machinist",
    "Librarian", "Psychologist", "Economist", "Statistician", "Professor",
    "Janitor", "Security Guard", "Driver", "Pilot", "Mechanic", "Veterinarian"
]


# --- Illustrative PII Components Data (English Alphabet Only) ---
# Transliterated names, street types, etc.
# --- Illustrative PII Components Data (English Alphabet Only) ---
# Transliterated names, street types, etc.
pii_components = {
    "Southern & Western Europe": {
        "Italy": {
            "first_names": ["Giovanni", "Maria", "Francesco", "Sofia", "Alessandro", "Giulia", "Antonio", "Anna", "Marco", "Luca", "Chiara", "Matteo", "Sara", "Davide", "Elena", "Paolo"],
            "last_names": ["Rossi", "Ferrari", "Russo", "Conti", "Bianchi", "Esposito", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Gallo", "Bruno", "Barbieri", "Lombardi", "Moretti"],
            "street_types": ["Via", "Piazza", "Corso", "Viale", "Largo", "Strada", "Vicolo", "Contrada", "Lungomare", "Salita"],
            "street_names": ["Roma", "Milano", "Napoli", "Venezia", "Garibaldi", "Cavour", "Manzoni", "Verdi", "Dante", "Petrarca", "Leopardi", "Mazzini", "Colombo", "Unita", "Europa", "Marconi", "Risorgimento", "Novembre", "Maggio", "Liberta"],
            "company_roots": ["Impresa", "Studio", "Societa", "Gruppo", "Officine", "Servizi", "Edilizia", "Finanza", "Costruzioni", "Tecnologie", "Soluzioni", "Logistica", "Manifattura", "Commercio", "Innovazione", "Progetti"],
            "company_suffixes": ["SRL", "SPA", "e Figli", "", "SNC", "SAS", "Azienda Agricola", "Cooperativa"],
            "bank_prefixes": ["Banca", "Credito", "Istituto di Credito", "Cassa di Risparmio", "Banco", "Unione di Banche"],
            "bank_roots": ["Italiano", "Regionale", "Popolare", "Centrale", "Finanziaria", "Nazionale", "Crediti", "Agricolo", "Commercio", "Sviluppo", "Risparmio", "Toscana", "Veneto", "Lombarda"],
            "bank_suffixes": ["Spa", "Gruppo", "", "Cooperativo", "Consorzio", "Nazionale"],
            "hospital_prefixes": ["Ospedale", "Clinica", "Policlinico", "Centro Medico", "Istituto", "Casa di Cura", "Presidio Ospedaliero", "Azienda Ospedaliera"],
            "hospital_roots": ["Civile", "Generale", "Regionale", "Universitario", "San Raffaele", "Sant'Andrea", "Madonnina", "Riuniti", "Fatebenefratelli", "Niguarda", "Spallanzani", "Bambino Gesu", "Cardarelli", "Pini"],
            "hospital_suffixes": ["", "SpA", "Fondazione", "IRCCS"],
        },
        "Spain": {
             "first_names": ["Manuel", "Sofia", "Javier", "Lucia", "Alejandro", "Maria", "David", "Laura", "Pablo", "Carmen", "Daniel", "Paula", "Adrian", "Elena", "Sergio", "Marta"],
             "last_names": ["Garcia", "Fernandez", "Lopez", "Martin", "Sanchez", "Gonzalez", "Rodriguez", "Perez", "Gomez", "Martinez", "Jimenez", "Ruiz", "Alonso", "Hernandez", "Diaz", "Moreno"],
             "street_types": ["Calle", "Plaza", "Avenida", "Paseo", "Carrera", "Ronda", "Camino", "Glorieta", "Travesia", "Bulevar"],
             "street_names": ["Mayor", "Sol", "Princesa", "Gran Via", "Colon", "Diagonal", "Castellana", "La Paz", "Cervantes", "Lopez de Vega", "Goya", "Velazquez", "Libertad", "Constitucion", "Independencia", "Real", "Nueva", "San Juan", "Reyes Catolicos", "America"],
             "company_roots": ["Empresa", "Grupo", "Estudio", "Servicios", "Consultoria", "Comercial", "Tecnica", "Soluciones", "Construcciones", "Internacional", "Industrial", "Inversiones", "Proyectos", "Desarrollo"],
             "company_suffixes": ["SL", "SA", "e Hijos", "", "S.Com.", "S.L.U.", "S.A.U.", "Cooperativa"],
             "bank_prefixes": ["Banco", "Caja", "Credito", "Banco Popular", "Caja Rural", "Banco de Ahorro"],
             "bank_roots": ["Espanol", "Nacional", "Comarcal", "Financiero", "Central", "Atlantico", "Popular", "Santander", "Bilbao", "Vizcaya", "Sabadell", "Andalucia", "Galicia", "Valencia"],
             "bank_suffixes": ["SA", "", "Grupo", "Entidad"],
             "hospital_prefixes": ["Hospital", "Clinica", "Centro Medico", "Sanatorio", "Instituto Medico", "Complejo Hospitalario"],
             "hospital_roots": ["General", "Universitario", "Provincial", "La Salud", "Quiron", "San Pablo", "La Fe", "Doce de Octubre", "Ramon y Cajal", "Gregorio Maranon", "Virgen del Rocio", "Carlos Haya", "Santa Creu", "Bellvitge"],
             "hospital_suffixes": ["", "SA", "SL", "Publico"],
        }
    },
    "Central & Western Europe": {
        "France": {
            "first_names": ["Jean", "Marie", "Pierre", "Sophie", "Antoine", "Camille", "Louis", "Chloe", "Lucas", "Manon", "Gabriel", "Louise", "Arthur", "Emma", "Hugo", "Alice"],
            "last_names": ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Petit", "Durand", "Leroy", "Moreau", "Simon", "Laurent", "Lefevre", "Roux", "Fournier", "Garcia", "Michel"],
            "street_types": ["Rue", "Avenue", "Boulevard", "Place", "Allee", "Impasse", "Chemin", "Route", "Quai", "Cours"],
            "street_names": ["Paris", "Lyon", "Marseille", "Liberte", "Republique", "Victor Hugo", "Pasteur", "Gambetta", "Clemenceau", "General de Gaulle", "Jean Jaures", "Jeanne dArc", "Foch", "Verdun", "Gare", "Eglise", "Chateau", "Moulin"],
            "company_roots": ["Societe", "Bureau", "Groupe", "Ateliers", "Companie", "Services", "Consultants", "Ingenierie", "Technologie", "Distribution", "Developpement", "Construction", "Financiere", "Solutions"],
            "company_suffixes": ["SARL", "SA", "et Fils", "", "SAS", "SNC", "EURL", "Cooperative"],
            "bank_prefixes": ["Banque", "Credit", "Societe", "Caisse dEpargne", "Comptoir National", "Banque Privee"],
            "bank_roots": ["National", "Regional", "Populaire", "Financier", "Mutuel", "Agricole", "Industriel", "France", "Paris", "Lyon", "Europeenne", "Transatlantique", "Commercial", "Investissement"],
            "bank_suffixes": ["SA", "", "Groupe", "Privee"],
            "hospital_prefixes": ["Hopital", "Clinique", "Centre Hospitalier", "Polyclinique", "Institut", "Maison Medicale"],
            "hospital_roots": ["General", "Universitaire", "Regional", "La Sante", "Saint-Louis", "Cochin", "Pitie-Salpetriere", "Bichat", "Necker", "Broussais", "Europeen Georges Pompidou", "Sainte Anne", "Val de Grace", "Civil"],
            "hospital_suffixes": ["", "SA", "Prive", "Universitaire"],
        },
        "Switzerland": {
             "first_names": ["Hans", "Ursula", "Peter", "Brigitte", "Martin", "Claudia", "Stefan", "Andrea", "Daniel", "Monika", "Christian", "Sandra", "Markus", "Nicole", "Michael", "Anna"],
             "last_names": ["Mueller", "Schneider", "Keller", "Weber", "Meier", "Huber", "Fischer", "Gautschi", "Baumann", "Frei", "Widmer", "Gerber", "Schmid", "Brunner", "Suter", "Wyss"],
             "street_types": ["Strasse", "Weg", "Gasse", "Platz", "Allee", "Rue", "Via", "Chemin", "Sentier", "Promenade"],
             "street_names": ["Bahnhof", "Haupt", "Dorf", "Berg", "Tal", "Muster", "Kirch", "Sonnen", "Zentral", "Seiden", "Bundesplatz", "Limmatquai", "Rue du Rhone", "Via Nassa", "Paradeplatz", "Marktgasse", "Rosenweg", "Lindenhof", "See", "Wald"],
             "company_roots": ["AG", "GmbH", "Group", "Technik", "Systeme", "Consulting", "Finanz", "Holding", "Pharma", "Solutions", "Services", "International", "Trading", "Management"],
             "company_suffixes": ["AG", "GmbH", "SA", "", "SARL", "Holding", "Partner", "International"], 
             "bank_prefixes": ["Bank", "Credit", "Raiffeisen", "Banque Cantonale", "Privatbank", "Hypothekarbank"],
             "bank_roots": ["Schweizerisch", "National", "Kantonal", "Finanz", "Union", "Zentral", "Alpin", "Zurich", "Geneve", "Bern", "Vaudois", "Lombard", "Odier", "Julius Baer"],
             "bank_suffixes": ["AG", "", "SA", "Gruppe"],
             "hospital_prefixes": ["Spital", "Klinik", "Gesundheitszentrum", "Hopital Cantonal", "Universitaetsklinik", "Privatklinik"],
            "hospital_roots": ["Allgemein", "Kantonsspital", "Universitaetsspital", "Hirslanden", "Bethesda", "Zurich", "Geneva", "Bern", "Basel", "Luzern", "Insel", "Cecil", "Beau Site", "Lindenhof"],
            "hospital_suffixes": ["AG", "", "SA", "Stiftung"],
        },
        "Germany": {
            "first_names": ["Thomas", "Andrea", "Michael", "Sabine", "Andreas", "Christine", "Stefan", "Claudia", "Christian", "Julia", "Alexander", "Nicole", "Markus", "Stefanie", "Daniel", "Anja"],
            "last_names": ["Schmidt", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann", "Schaefer", "Koch", "Bauer", "Richter", "Klein", "Wolf", "Schroeder", "Neumann"],
            "street_types": ["Strasse", "Weg", "Allee", "Platz", "Gasse", "Ring", "Damm", "Ufer", "Promenade", "Chaussee"],
            "street_names": ["Haupt", "Bahnhof", "Berg", "Wald", "Garten", "Goethe", "Schiller", "Friedrich", "Karl", "Mittel", "Kirch", "Schul", "Linden", "Birken", "Eichen", "Markt", "Post", "Rosen", "Sonnen", "Nord"],
            "company_roots": ["GmbH", "AG", "Werke", "Systeme", "Consulting", "Industrie", "Service", "Technologie", "Handel", "Bau", "Finanz", "Logistik", "Automobil", "Energie"],
            "company_suffixes": ["GmbH", "AG", "KG", "", "und Co KG", "eG", "Stiftung", "KGaA"],
            "bank_prefixes": ["Bank", "Sparkasse", "Volksbank", "Deutsche", "Commerzbank", "HypoVereinsbank"],
            "bank_roots": ["Deutsche", "Nationale", "Sparkasse", "Volksbank", "Landesbank", "Kommunal", "Hanseatisch", "Berliner", "Frankfurter", "Hamburger", "Bayerische", "Mittelstand", "Direkt", "Investitions"],
            "bank_suffixes": ["AG", "", "eG", "KGaA", "und Co KG"],
            "hospital_prefixes": ["Krankenhaus", "Klinik", "Universitaetsklinikum", "Staedtisches Klinikum", "Fachklinik", "Bezirksklinikum"],
            "hospital_roots": ["Allgemein", "Staedtisch", "Universitaets", "Diakonie", "Marien", "Evangelisch", "Westend", "Charite", "Heidelberg", "Muenchen", "Hamburg Eppendorf", "Rechts der Isar", "Sankt Georg", "Elisabeth"],
            "hospital_suffixes": ["gGmbH", "AG", "", "Stiftung", "Klinikum", "Zentrum"],
        },
        "Netherlands": {
            "first_names": ["Jan", "Maria", "Piet", "Anna", "Hendrik", "Cornelia", "Dirk", "Elizabeth", "Johannes", "Wilhelmina", "Willem", "Johanna", "Kees", "Grietje", "Gerrit", "Neeltje"],
            "last_names": ["Jansen", "De Vries", "Bakker", "Van Dijk", "Smit", "De Jong", "Willems", "Peters", "Visser", "Bos", "Mulder", "Van den Berg", "Dekker", "Brouwer", "Jacobs", "Vermeulen"],
            "street_types": ["Straat", "Weg", "Laan", "Plein", "Gracht", "Dijk", "Singel", "Kade", "Steeg", "Pad", "Hof", "Markt"],
            "street_names": ["Dorp", "Kerk", "Molen", "Nieuw", "Oud", "Linden", "Beuken", "Eiken", "Park", "Kanaal", "Hoofd", "Voor", "Achter", "School", "Binnen", "Buiten", "Oranje", "Prins Hendrik", "Julian", "Wilhelmina"],
            "company_roots": ["BV", "NV", "Groep", "Advies", "Techniek", "Service", "Handel", "Holding", "Solutions", "Consulting", "International", "Bouw", "Transport", "Media"],
            "company_suffixes": ["BV", "NV", "VOF", "", "Holding", "Groep", "International", "Nederland"],
            "bank_prefixes": ["Bank", "Rabobank", "ING", "ABN AMRO"],
            "bank_roots": ["Nederlandse", "Regionale", "Cooperatieve", "Financieel", "Spaar", "Handels", "Volks", "Hypotheek", "Effecten", "Trust"],
            "bank_suffixes": ["NV", "", "Bank", "Groep"],
            "hospital_prefixes": ["Ziekenhuis", "Kliniek", "Medisch Centrum", "Academisch Medisch Centrum", "Universitair Medisch Centrum", "Streekziekenhuis"],
            "hospital_roots": ["Algemeen", "Regionaal", "Universitair", "Sint", "Groene Hart", "Academisch", "Stads", "Onze Lieve Vrouwe", "Antoni van Leeuwenhoek", "Erasmus", "LUMC", "Maastricht UMC", "Radboud", "Vrije Universiteit"],
            "hospital_suffixes": ["", "BV", "NV", "Stichting"],
        }
    },
    "Nordic Countries": {
        "Sweden": {
            "first_names": ["Lars", "Ingrid", "Anders", "Elisabeth", "Johan", "Christina", "Erik", "Sofia", "Mikael", "Anna", "Per", "Eva", "Karl", "Maria", "Daniel", "Emma"],
            "last_names": ["Andersson", "Johansson", "Karlsson", "Nilsson", "Eriksson", "Larsson", "Olsson", "Svensson", "Persson", "Gustafsson", "Pettersson", "Jonsson", "Holm", "Berg", "Lindberg", "Nyberg"],
            "street_types": ["Vagen", "Gatan", "Grand", "Torget", "Allen", "Stigen", "Leden", "Parken", "Stranden", "Bryggan"],
            "street_names": ["Storgatan", "Kyrkogatan", "Skogs", "Bergs", "Central", "Station", "Norra", "Sodra", "Kungsgatan", "Drottninggatan", "Vasagatan", "Ostra", "Vastra", "Industri", "Skol", "Hamn", "Strand", "Ring", "Park"],
            "company_roots": ["AB", "Gruppen", "Konsult", "Teknik", "Service", "Handel", "System", "Solutions", "Industri", "Partner", "Nordic", "Data"],
            "company_suffixes": ["AB", "", "HB", "KB", "Ek for"],
            "bank_prefixes": ["Bank", "Sparbank", "Handelsbanken", "Nordea", "SEB", "Swedbank"],
            "bank_roots": ["Svenska", "Nationella", "Lokala", "Finans", "Spar", "Hypotek", "Foretags", "Privat", "Investment", "Nordic"],
            "bank_suffixes": ["AB", "", "ASA", "Stadshypotek"],
            "hospital_prefixes": ["Sjukhus", "Klinik", "Vardcentral", "Lasarett", "Region", "Akademiska"],
            "hospital_roots": ["Allmanna", "Lans", "Universitets", "Karolinska", "St Gorans", "Central", "Stads", "Sahlgrenska", "Akademiska", "Danderyds", "Orebro", "Uppsala", "Linkoping", "Norrlands"],
            "hospital_suffixes": ["AB", "", "Region", "Landstinget"],
        },
        "Norway": {
            "first_names": ["Ole", "Hege", "Bjorn", "Anne", "Morten", "Camilla", "Espen", "Line", "Jan", "Inger", "Kjetil", "Marianne", "Thomas", "Hilde", "Per", "Silje"],
            "last_names": ["Hansen", "Jensen", "Kristiansen", "Andersen", "Pedersen", "Nilsen", "Eriksen", "Berg", "Larsen", "Johansen", "Olsen", "Solberg", "Bakken", "Moen", "Lien", "Andreassen"],
            "street_types": ["Veien", "Gata", "Plassen", "Gate", "Alleen", "Stien", "Bryggen", "Kaia", "Torget", "Kroken"],
            "street_names": ["Hoved", "Kirke", "Skole", "Park", "Sentrums", "Bygdoy", "Frogner", "Grunerlokka", "Storgata", "Karl Johans", "Drammensveien", "Slottsplassen", "Radhusplassen", "Aker Brygge", "Industrigata", "Fjordveien"],
            "company_roots": ["AS", "Gruppen", "Konsulent", "Teknikk", "Service", "Handel", "Systems", "Solutions", "Industri", "Partner", "Maritime", "Holding"],
            "company_suffixes": ["AS", "ASA", "", "Holding", "Consulting", "Group"],
            "bank_prefixes": ["Bank", "Sparebank", "DNB", "Nordea"],
            "bank_roots": ["Norske", "Regionale", "Spare", "Finans", "Kommune", "Handels", "Kreditt", "Landbruks", "Sjoefart", "Bolig"],
            "bank_suffixes": ["ASA", "", "Bank", "Gruppe"],
            "hospital_prefixes": ["Sykehus", "Klinikk", "Helsestasjon", "Distriktsmedisinsk senter", "Spesialistsykehus", "Universitetssykehus"],
            "hospital_roots": ["Generelle", "Fylkes", "Universitets", "Ulleval", "Haukeland", "Regions", "Sentral", "Rikshospitalet", "Aker", "St Olavs", "Tromso", "Stavanger", "Bergen", "Oslo"],
            "hospital_suffixes": ["HF", "", "AS", "Stiftelse"],
        },
         "Denmark": {
            "first_names": ["Jens", "Anne", "Lars", "Bente", "Mads", "Hanne", "Peter", "Mette", "Christian", "Kirsten", "Michael", "Lone", "Henrik", "Susanne", "Soren", "Camilla"],
            "last_names": ["Jensen", "Nielsen", "Hansen", "Pedersen", "Andersen", "Christensen", "Larsen", "Sorensen", "Rasmussen", "Jorgensen", "Petersen", "Madsen", "Kristensen", "Olsen", "Thomsen", "Poulsen"],
            "street_types": ["Vej", "Gade", "Plads", "Alle", "Park", "Sti", "Torv", "Boulevard", "Kaj", "Bro"],
            "street_names": ["Hoved", "Kirke", "Skole", "By", "Strand", "Tivoli", "Nyhavn", "Raadhus", "Ostergade", "Vestergade", "Nygade", "Kongens Nytorv", "Amagertorv", "Bredgade", "Gammel", "Store"],
            "company_roots": ["A/S", "Gruppen", "Konsulent", "Teknik", "Service", "Handel", "Systemer", "Losninger", "Industri", "Partner", "Holding", "Design"],
            "company_suffixes": ["A/S", "ApS", "", "Holding", "International", "Group"],
            "bank_prefixes": ["Bank", "Sparekasse", "Danske Bank", "Nordea", "Jyske Bank"],
            "bank_roots": ["Danske", "Nationale", "Lokale", "Finans", "Sparekasse", "Arbejdernes", "Hypotek", "Landbobank", "Kredit", "Alm Brand"],
            "bank_suffixes": ["A/S", "", "Bank", "Fondsmæglerselskab"],
            "hospital_prefixes": ["Hospital", "Klinik", "Sundhedscenter", "Privathospital", "Regionshospital", "Universitetshospital"],
            "hospital_roots": ["Almindelige", "Regions", "Universitets", "Rigshospitalet", "Aarhus", "Kobenhavns", "Central", "Odense", "Aalborg", "Herlev", "Gentofte", "Skejby", "Hvidovre", "Bispebjerg"],
            "hospital_suffixes": ["", "A/S", "Region", "Center"],
        },
         "Finland": {
            "first_names": ["Matti", "Liisa", "Timo", "Anna", "Jari", "Satu", "Antti", "Johanna", "Juha", "Pirjo", "Kari", "Ritva", "Mikko", "Paivi", "Pekka", "Leena"],
            "last_names": ["Korhonen", "Nieminen", "Makinen", "Virtanen", "Jarvinen", "Laine", "Hamalainen", "Koskinen", "Heikkinen", "Lehtonen", "Saarinen", "Kallio", "Rantanen", "Pitkanen", "Salminen", "Lehtinen"],
            "street_types": ["Tie", "Katu", "Polku", "Aukio", "Kuja", "Ranta", "Silta", "Tori", "Puisto", "Kaari"],
            "street_names": ["Kirkko", "Kauppa", "Maki", "Jarvi", "Metsa", "Koivu", "Vaahtera", "Linna", "Keskus", "Asema", "Rautatie", "Satama", "Koulu", "Uusi", "Vanha", "Teollisuus"],
            "company_roots": ["Oy", "Konsultointi", "Tekniikka", "Palvelu", "Rakennus", "Jarjestelmat", "Ratkaisut", "Teollisuus", "Kumppani", "Holding", "Design", "Logistiikka"],
            "company_suffixes": ["Oy", "Ltd", "", "Oyj", "Holding", "Group"],
            "bank_prefixes": ["Pankki", "Saastopankki", "Osuuspankki", "Nordea", "Danske Bank"],
            "bank_roots": ["Suomen", "Kansallinen", "Paikallinen", "Finanssi", "Spaar", "Hypoteekki", "Yritys", "Sijoitus", "Aktia", "POP"],
            "bank_suffixes": ["Oy", "", "Oyj", "Asuntoluottopankki"],
            "hospital_prefixes": ["Sairaala", "Klinikka", "Terveyskeskus", "Yliopistollinen sairaala", "Keskussairaala", "Aluesairaala"],
            "hospital_roots": ["Yleinen", "Alue", "Yliopisto", "Helsingin", "Tampereen", "Keskus", "Kaupungin", "Turun", "Oulun", "Kuopion", "Mehilainen", "Terveystalo", "Pohjois", "Etela"],
            "hospital_suffixes": ["Oy", "", "Oyj", "Kuntayhtyma"],
        }
    },
    "Anglophone Countries": {
        "United Kingdom": {
            "first_names": ["John", "Mary", "David", "Sarah", "James", "Elizabeth", "Michael", "Jessica", "William", "Susan", "Robert", "Linda", "Richard", "Karen", "Thomas", "Patricia"],
            "last_names": ["Smith", "Jones", "Williams", "Brown", "Taylor", "Wilson", "Johnson", "Davies", "Evans", "Roberts", "Walker", "Wright", "Thompson", "White", "Green", "Hall"],
            "street_types": ["Street", "Road", "Lane", "Avenue", "Close", "Drive", "Gardens", "Way", "Crescent", "Place", "Court", "Grove", "Hill", "Mews"],
            "street_names": ["High", "Main", "Park", "Church", "Victoria", "King", "Queen", "Station", "Green", "Mill", "London", "Oxford", "Cambridge", "York", "School", "Manor", "Orchard", "New", "Old", "Bridge"],
            "company_roots": ["Ltd", "Plc", "Group", "Solutions", "Systems", "Consulting", "Engineering", "Services", "Holdings", "Ventures", "Associates", "Partners", "Global", "Technologies"],
            "company_suffixes": ["Ltd", "Plc", "LLP", "", "Limited", "and Sons", "Group", "International"],
            "bank_prefixes": ["Bank of", "National", "Royal", "Lloyds", "Barclays", "HSBC", "Santander", "TSB", "Metro Bank", "Clydesdale Bank"],
            "bank_roots": ["British", "County", "Global", "Capital", "Midland", "National", "Scottish", "Irish", "London", "Manchester", "Commercial", "Savings"],
            "bank_suffixes": ["Plc", "Group", "", "UK", "Limited", "Banking Group"],
            "hospital_prefixes": ["St.", "General", "Royal", "City", "County", "University", "NHS Trust", "Spire"],
            "hospital_roots": ["County", "Teaching", "Community", "King Edward", "Queen Mary", "Victoria", "Central", "London", "Manchester Royal", "Addenbrookes", "John Radcliffe", "Guys and St Thomas", "Great Ormond Street", "Birmingham"],
            "hospital_suffixes": ["Hospital", "Clinic", "Medical Centre", "Infirmary", "Trust", "Foundation Trust"],
        },
        "US": {
            "first_names": ["Michael", "Jennifer", "David", "Jessica", "Christopher", "Ashley", "Matthew", "Amanda", "James", "Sarah", "Robert", "Melissa", "John", "Nicole", "William", "Stephanie"],
            "last_names": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas"],
            "street_types": ["Street", "Avenue", "Road", "Lane", "Drive", "Blvd", "Court", "Place", "Terrace", "Way", "Circle", "Highway", "Pike", "Trail", "Expressway", "Freeway"],
            "street_names": ["Main", "Oak", "Pine", "Maple", "Cedar", "Park", "Washington", "Franklin", "Broadway", "Elm", "Lincoln", "Madison", "Jefferson", "Adams", "Monroe", "State", "First", "Second", "Third", "Market"],
            "company_roots": ["Corp", "Inc", "Group", "Solutions", "Systems", "Enterprises", "Holdings", "Industries", "Technologies", "Services", "Global", "National", "American", "International"],
            "company_suffixes": ["Inc.", "Corp.", "LLC", "", "Ltd.", "Co.", "Group", "Holdings"],
            "bank_prefixes": ["Bank of", "First National", "Chase", "Wells Fargo", "Citibank", "US Bank"],
            "bank_roots": ["American", "State", "Capital", "Community", "Federal", "Union", "Citizens", "National", "Commerce", "Trust", "Security", "Peoples", "Savings", "United"],
            "bank_suffixes": ["N.A.", "Group", "", "Bank", "Corp", "Company"],
            "hospital_prefixes": ["St.", "General", "Mercy", "Community", "Memorial", "University", "Medical", "Kaiser Permanente", "HCA", "Providence"],
            "hospital_roots": ["County", "Memorial", "University", "Methodist", "Baptist", "Presbyterian", "City", "Regional", "Medical Center", "Childrens", "General", "Health", "North", "South"],
            "hospital_suffixes": ["Hospital", "Medical Center", "Clinic", "Healthcare", "System", "Campus"],
        },
        "Canada": {
             "first_names": ["Michael", "Jennifer", "David", "Sarah", "James", "Elizabeth", "Robert", "Mary", "William", "Linda", "Christopher", "Patricia", "Daniel", "Susan", "Matthew", "Jessica"],
             "last_names": ["Smith", "Jones", "Williams", "Brown", "Taylor", "Wilson", "Miller", "Davis", "Tremblay", "Martin", "Roy", "Gagnon", "Lee", "Johnson", "McDonald", "Campbell"],
             "street_types": ["Street", "Road", "Avenue", "Crescent", "Place", "Boulevard", "Trail", "Drive", "Way", "Court", "Line", "Route", "Gardens", "Terrace"],
             "street_names": ["Main", "Centre", "Park", "Church", "Victoria", "King", "Queen", "Bay", "Yonge", "Ste-Catherine", "University", "College", "West", "East", "North", "South", "Rue Principale", "First", "Second", "Maple"],
             "company_roots": ["Inc", "Corp", "Group", "Solutions", "Systems", "Enterprises", "Holdings", "Industries", "Technologies", "Services", "Canadian", "National", "Global", "Ventures"],
             "company_suffixes": ["Inc.", "Corp.", "Ltd.", "", "Ltee.", "Limited", "Group", "International"],
             "bank_prefixes": ["Bank of", "National", "Royal Bank", "TD", "Scotiabank", "BMO", "CIBC", "Desjardins", "HSBC Bank", "Laurentian Bank"],
             "bank_roots": ["Canadian", "Provincial", "Commerce", "National", "Federal", "Dominion", "Montreal", "Nova Scotia", "Toronto", "Imperial", "Trust", "Credit Union"],
             "bank_suffixes": ["", "Inc.", "Group", "Canada", "Financial"],
             "hospital_prefixes": ["St.", "General", "Royal", "City", "University Health", "Mount Sinai", "Health Sciences", "Regional"], 
             "hospital_roots": ["Provincial", "Civic", "University", "Toronto General", "Vancouver General", "Montreal General", "Royal Victoria", "SickKids", "Foothills Medical", "Ottawa Hospital", "Hamilton Health", "Kingston General", "Sunnybrook", "Credit Valley"],
             "hospital_suffixes": ["Hospital", "Medical Centre", "Clinic", "Health Centre", "Institute", "Foundation"],
        },
        "Australia": {
             "first_names": ["Michael", "Sarah", "David", "Jessica", "James", "Emily", "Matthew", "Elizabeth", "William", "Olivia", "Lachlan", "Chloe", "Daniel", "Sophie", "Chris", "Isabella"],
             "last_names": ["Smith", "Jones", "Williams", "Brown", "Taylor", "Wilson", "Kelly", "Ryan", "Walker", "Harris", "Thompson", "Lee", "Martin", "Anderson", "White", "Nguyen"],
             "street_types": ["Street", "Road", "Avenue", "Crescent", "Close", "Lane", "Parade", "Place", "Drive", "Way", "Court", "Esplanade", "Highway", "Terrace"],
             "street_names": ["High", "Main", "Park", "Church", "Victoria", "King", "Queen", "Oxford", "George", "Pitt", "Collins", "Elizabeth", "Macquarie", "William", "Flinders", "Swanston", "North", "South", "East", "West"],
             "company_roots": ["Pty Ltd", "Group", "Solutions", "Systems", "Consulting", "Services", "Holdings", "Ventures", "National", "Australian", "Pacific", "Mining", "Resources", "Industries"],
             "company_suffixes": ["Pty Ltd", "Ltd", "", "Group", "Holdings", "No Liability", "NL"],
             "bank_prefixes": ["Bank of", "National", "Commonwealth Bank", "ANZ", "Westpac", "NAB", "Bendigo Bank", "Bankwest", "Suncorp"],
             "bank_roots": ["Australian", "State", "Regional", "Westpac", "National", "Queensland", "South Australia", "New South Wales", "Victoria", "Tasmania", "Capital", "Investment"],
             "bank_suffixes": ["Ltd", "", "Bank", "Group", "Limited"],
             "hospital_prefixes": ["St.", "General", "Royal", "City", "Public", "Private", "Community", "District"],
             "hospital_roots": ["State", "Base", "Teaching", "Sydney Hospital", "Royal Melbourne", "Prince Alfred", "Princess Alexandra", "Royal North Shore", "Alfred Hospital", "Monash Medical", "Fiona Stanley", "Westmead", "Womens and Childrens", "Mater"],
             "hospital_suffixes": ["Hospital", "Medical Centre", "Clinic", "Health Service", "Campus", "Network"],
        },
         "New Zealand": {
             "first_names": ["Michael", "Sarah", "David", "Jessica", "James", "Emily", "Daniel", "Hannah", "William", "Olivia", "Joshua", "Sophie", "Samuel", "Chloe", "Benjamin", "Isabella"],
             "last_names": ["Smith", "Jones", "Williams", "Brown", "Taylor", "Wilson", "Scott", "Anderson", "Thompson", "Walker", "Clark", "Young", "Miller", "Harris", "White", "Campbell"],
             "street_types": ["Street", "Road", "Avenue", "Crescent", "Close", "Terrace", "Place", "Drive", "Lane", "Way", "Grove", "Parade"],
             "street_names": ["High", "Main", "Park", "Church", "Victoria", "King", "Queen", "George", "Princes", "Willis", "Lambton Quay", "Colombo", "Albert", "Dominion", "Beach", "Station", "Cambridge", "Richmond", "Nelson", "Grey"],
             "company_roots": ["Ltd", "Group", "Solutions", "Systems", "Consulting", "Services", "Holdings", "Ventures", "National", "New Zealand", "Pacific", "Enterprises", "Developments", "Technologies"],
             "company_suffixes": ["Ltd", "", "Limited", "Group", "Holdings", "NZ"],
             "bank_prefixes": ["Bank of", "National", "ANZ", "ASB Bank", "BNZ", "Kiwibank", "Westpac", "TSB Bank"],
             "bank_roots": ["New Zealand", "Regional", "Kiwibank", "National", "Trust", "South", "Heartland", "Cooperative", "Savings", "Investment"],
             "bank_suffixes": ["Ltd", "", "Bank", "Limited", "Group"],
             "hospital_prefixes": ["St.", "General", "Royal", "City", "District Health", "Public", "Community", "Memorial"],
             "hospital_roots": ["Regional", "Base", "Teaching", "Auckland City", "Wellington", "Christchurch", "Dunedin", "Middlemore", "Waikato", "North Shore", "Starship", "Palmerston North", "Tauranga", "Nelson Marlborough"],
             "hospital_suffixes": ["Hospital", "Medical Centre", "Clinic", "Health", "Board", "Campus"],
        }
    },
    "East Asia": {
        "Japan": {
            "first_names": ["Hiroshi", "Yuko", "Takashi", "Ayumi", "Kenji", "Sakura", "Takeshi", "Akiko", "Taro", "Hanako", "Ichiro", "Haruka", "Jiro", "Mei", "Kazuo", "Naomi"],
            "last_names": ["Tanaka", "Sato", "Suzuki", "Takahashi", "Watanabe", "Ito", "Yamamoto", "Nakamura", "Kobayashi", "Kato", "Yoshida", "Yamada", "Sasaki", "Matsumoto", "Inoue", "Kimura"],
            "street_types": ["Chome", "Ban", "Go", "Machi", "Jima", "Dori", "Ku", "Shi", "Gun", "Son"], # District, City, County, Village etc.
            "street_names": ["Ginza", "Shinjuku", "Shibuya", "Marunouchi", "Akasaka", "Aoyama", "Kanda", "Roppongi", "Ueno", "Asakusa", "Chuo", "Minato", "Taito", "Sumida", "Otemachi", "Nihonbashi", "Ikebukuro", "Shinagawa", "Nakano", "Suginami"],
            "company_roots": ["Kabushiki Kaisha", "Yugen Kaisha", "Godo Kaisha", "Sangyo", "Denki", "Kogyo", "Shoji", "Consulting", "Jitsugyo", "Kaihatsu", "System", "Engineering", "Network", "Holdings", "International", "Electronics"],
            "company_suffixes": ["K.K.", "Co., Ltd.", "G.K.", "", "Ltd.", "Inc.", "Corp.", "Japan"],
            "bank_prefixes": ["Bank of", "Japan", "Industrial Bank of", "Sumitomo Mitsui", "Mizuho", "Resona"],
            "bank_roots": ["Nippon", "Tokyo", "Sumitomo", "Mizuho", "MUFG", "Sakura", "Fuji", "Chuo", "Daiwa", "Yokohama", "Chiba", "Shizuoka", "Kyoto", "Hiroshima"],
            "bank_suffixes": ["", "Limited", "Bank", "Trust", "Financial Group"],
            "hospital_prefixes": ["Byoin", "Iin", "Medical Center", "Sogo Byoin", "Daigaku Byoin", "Clinic"], # Hospital, Clinic, General Hospital, University Hospital
            "hospital_roots": ["Daiichi", "Daini", "Central", "University", "City", "Prefectural", "National", "Tokyo", "Osaka", "Kyoto University", "Keio University", "Juntendo", "Red Cross", "Saiseikai"],
            "hospital_suffixes": ["", "Byoin", "Iryo Center", "Foundation"],
        },
        "South Korea": {
            "first_names": ["Ji-hoon", "Seo-yeon", "Min-jun", "Da-eun", "Sung-hyun", "Ha-eun", "Joon-ho", "Yeon-woo", "Do-yun", "Seo-yun", "Hyun-woo", "Ji-woo", "Chul-soo", "Young-hee", "Sang-chul", "Mi-kyung"],
            "last_names": ["Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoo", "Yoon", "Jang", "Lim", "Han", "Oh", "Shin", "Seo", "Kwon"],
            "street_types": ["gil", "ro", "dong", "daero", "ga", "eup", "myeon"], # Street, Road, Neighborhood, Boulevard, Town, Township
            "street_names": ["Gangnam", "Myeongdong", "Hongdae", "Itaewon", "Jongno", "Insadong", "Sinchon", "Apgujeong", "Teheran", "Euljiro", "Sejong", "Yoido", "Mapo", "Seocho", "Songpa", "Namsan"],
            "company_roots": ["Jusik Hoesa", "Yuhan Hoesa", "Gongdong Hoesa", "Sanup", "Jeonja", "Trading", "Consulting", "Systems", "Electronics", "Heavy Industries", "Chemical", "Construction", "Telecommunication", "Solutions"],
            "company_suffixes": ["Co., Ltd.", "", "Inc.", "Corp.", "Group", "Korea"], 
            "bank_prefixes": ["Bank of", "Korea", "Industrial Bank of", "Kookmin", "Shinhan", "Hana"],
            "bank_roots": ["Hana", "Shinhan", "Woori", "KB Kookmin", "National", "Central", "Nonghyup", "Suhyup", "Busan", "Daegu", "Kwangju", "Jeonbuk", "Kyongnam", "Development"],
            "bank_suffixes": ["Bank", "", "Financial Group", "Ltd"],
            "hospital_prefixes": ["Byeongwon", "Uiwon", "Medical Center", "Daehak Byeongwon", "Jonghap Byeongwon", "Clinic"], # Hospital, Clinic, University Hospital, General Hospital
            "hospital_roots": ["Central", "University", "Samsung", "Asan", "City", "National", "General", "Seoul National University", "Yonsei Severance", "Korea University", "Catholic University", "Hanyang University", "Kyung Hee University", "Chung Ang University"],
            "hospital_suffixes": ["", "Byeongwon", "Medical Foundation", "Healthcare System"],
        }
    },
     "Eastern Europe": {
        "Russia": {
            "first_names": ["Ivan", "Elena", "Sergei", "Anna", "Vladimir", "Natalia", "Dmitri", "Olga", "Alexander", "Svetlana", "Mikhail", "Tatiana", "Alexei", "Maria", "Nikolai", "Irina"],
            "last_names": ["Ivanov", "Petrov", "Smirnov", "Sokolov", "Kozlov", "Novikov", "Morozov", "Volkov", "Popov", "Lebedev", "Semenov", "Egorov", "Pavlov", "Mikhailov", "Fedorov", "Orlov"],
            "street_types": ["Ulitsa", "Prospekt", "Pereulok", "Ploshchad", "Bulvar", "Shosse", "Naberezhnaya", "Proezd", "Tupik", "Doroga"], # Street, Avenue, Lane, Square, Boulevard, Highway, Embankment, Passage, Dead-end, Road
            "street_names": ["Tverskaya", "Nevsky", "Arbat", "Gogol", "Lenin", "Mira", "Pushkin", "Gorky", "Kremlin", "Moskovsky", "Sadovaya", "Pervomayskaya", "Sovetskaya", "Kutuzovsky", "Leningradsky", "Komsomolsky", "Oktyabrskaya", "Lesnaya", "Polevoy", "Zelenaya"],
            "company_roots": ["OOO", "AO", "PAO", "Torgovy Dom", "Promyshlenny", "Service", "Trading", "Stroitelny", "Investitsionny", "Nauchno Proizvodstvenny", "Holding", "Gruppa Kompaniy", "Konsalting", "Tekhnologii"],
            "company_suffixes": ["OOO", "AO", "PAO", "", "Gruppa", "Holding", "Tsentr", "Kombinat"],
            "bank_prefixes": ["Bank", "Sberbank", "VTB", "Gazprombank", "Alfa Bank", "Rosselkhozbank", "Promsvyazbank", "Otkritie"],
            "bank_roots": ["Rossiyskiy", "Natsionalny", "Regionalny", "Finansovy", "Centralny", "Industrialny", "Moskovskiy", "Sibirskiy", "Uralskiy", "Investitsionny", "Kommercheskiy", "Narodny"],
            "bank_suffixes": ["", "PAO", "AO", "Bank", "Gruppa"],
            "hospital_prefixes": ["Bolnitsa", "Poliklinika", "Meditsinskiy Tsentr", "Klinicheskaya Bolnitsa", "Gorodskaya Bolnitsa", "Detskaya Bolnitsa"], # Hospital, Polyclinic, Medical Center, Clinical Hospital, City Hospital, Childrens Hospital
            "hospital_roots": ["Gorodskaya", "Oblastnaya", "Klinicheskaya", "Centralnaya", "Universitetskaya", "Detskaya", "Voennaya", "Skoroy Pomoshchi", "Pervaya", "Imeni Botkina", "Imeni Sklifosovskogo", "Regionalnaya", "Respublikanskaya", "Mediko Sanitarnaya Chast"],
            "hospital_suffixes": ["", "Bolnitsa", "Tsentr", "Klinika"],
        }
    }
}

