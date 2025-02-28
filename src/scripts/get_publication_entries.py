from bs4 import BeautifulSoup
import pandas as pd
import requests
import csv
from scholarly import scholarly


def get_publication_data(queries, num_results):
    all_data = []
    for query in queries:
        search_query = scholarly.search_pubs(query)
        for i in range(num_results):
            try:
                pub = next(search_query)
                bib = pub['bib']
                data = {
                    'authors': ", ".join(bib.get('author', [])),
                    'title': bib.get('title', ''),
                    'snippet': bib.get('abstract', ''),
                    'link': pub.get('pub_url', ''),
                    'publication_series': bib.get('venue', ''),
                    'publisher': bib.get('publisher', ''),
                    'year': bib.get('pub_year', ''),
                    'search_keys': search_query
                }
                all_data.append(data)
            except StopIteration:
                break

    return all_data

def create_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Exagerations

queries_from_citations_txt = [ "Shapiro  , Susan  Agency theory", \
                                "Panda    , Brahmadev                   , Nabaghan Madhabika; 	Agency theory: Review of theory and evidence on problems and perspectives", \
                                "Bendickson   , Josh                 , Jeff; Liguori, Eric; Davis, Phillip E; 	Agency theory: the times, they are a-changin’", \
                                "Heath    , Joseph                 uses and abuses of agency theory", \
                                "Ballwieser   , Wolfgang                 , G; Beckmann, MJ; Bester, H; Blickle, M; Ewert, R; Feichtinger, G; Firchau, V; Fricke, F; Funke, H; 	Agency theory, information, and incentives", \
                                "Shogren  , Karrie              Wehmeyer, Michael L; Palmer, Susan B; 	Causal agency theory", \
                                "Bendickson   , Josh                 , Jeff; Liguori, Eric W; Davis, Phillip E; 	Agency theory: background and epistemology", \
                                "Gallagher    , Shaun                  natural philosophy of agency", \
                                "Bai  , Heesoon                       for education: Towards human agency", \
                                "Taylor   , Charles                  agency and language", \
                                "Roessler     , Johannes               , Naomi; 	Agency and self-awareness: Issues in philosophy and psychology", \
                                "Hornsby  , Jennifer                  and actions", \
                                "Seligman     , Martin                    in greco-roman philosophy", \
                                "Proust   , Joëlle                 philosophy of metacognition: Mental agency and self-awareness", \
                                "Jurist   , Elliot              	Beyond Hegel and Nietzsche: Philosophy, culture, and agency", \
                                "Reader   , Soran                  other side of agency", \
                                "Splitter     , Laurance            	Agency, thought, and language: Analytic philosophy goes to school", \
                                "Stapleton    , Mog                 , Tom; 	The enactive philosophy of embodiment: From biological foundations of agency to the phenomenology of subjectivity", \
                                "Bhaskar  , Roy                philosophy of meta-reality: Part II: Agency, perfectibility, novelty", \
                                "Duff     , Robin                    	Intention, agency and criminal liability: Philosophy of action and the criminal law", \
                                "Watson   , Gary                agency", \
                                "Bayne    , Tim                phenomenology of agency", \
                                "Vargas   , Manuel                 , Gideon; 	Rational and social agency: the philosophy of Michael Bratman", \
                                "Seok     , Bongrae                  agency, autonomy, and heteronomy in early Confucian philosophy", \
                                "Thalberg     , Irving                     of agency: Studies in the philosophy of human action", \
                                "Sen  , Amartya                   and agency", \
                                "Dasti    , Matthew             Bryant, Edwin F; 	Free will, agency, and selfhood in Indian philosophy", \
                                "Castro   -Toledo,                           ; Cerezo, Pablo; Gómez-Bellvís, Ana Belén; 	Scratching the structure of moral agency: insights from philosophy applied to neuroscience", \
                                "Giordano     , James                    agency in pain medicine: Philosophy, practice and virtue", \
                                "Schlosser    , Markus              	Agency, ownership, and the standard theory", \
                                "Duff     , RA                      , Agency and Criminal Liability: Philosophy of Action and the Criminal Law (Introduction)", \
                                "Chopra   , Samir                  , Laurence; 	Artificial agents-personhood in law and philosophy", \
                                "Pippin   , Robert                     practical philosophy", \
                                "List     , Christian                   , Philip; 	Group agency: The possibility, design, and status of corporate agents", \
                                "Madden   , Edward              	Commonsense and agency theory", \
                                "Sebo     , Jeff                  and moral status", \
                                "Taylor   , Charles                          papers: Volume 1, Human agency and language", \
                                "Ellis    , Brian                  power of agency", \
                                "Emirbayer    , Mustafa                 , Ann; 	What is agency?", \
                                "Cunningham   , Stanley             	Reclaiming moral agency: the moral philosophy of Albert the Great", \
                                "Kelz     , Rosine                 non-sovereign self, responsibility, and otherness: Hannah Arendt, Judith Butler, and Stanley Cavell on moral philosophy and political agency", \
                                "Callinicos   , Alex                  history: Agency, structure, and change in social theory", \
                                "Earley   , Joseph                 -organization and agency: in chemistry and in process philosophy", \
                                "Korsgaard    , Christine               	Self-constitution: Agency, identity, and integrity", \
                                "Belnap   , Nuel                     and forwards in the modal logic of agency", \
                                "Lotto    , Michelle            Banoub, Mark; Schubert, Armin; 	Effects of anesthetic agents and physiologic changes on intraoperative motor evoked potentials", \
                                "Sies     , Helmut                 , Dean P; 	Reactive oxygen species (ROS) as pleiotropic physiological signalling agents", \
                                "Cannon   , Walter              	Organization for physiological homeostasis", \
                                "Bérubé   , Marie                   , Meenakshi; Hall, Dennis G; 	Benzoboroxoles as efficient glycopyranoside-binding agents in physiological conditions: structure and selectivity of complex formation", \
                                "Fredholm     , Bertil              	Adenosine—a physiological or pathophysiological agent?", \
                                "Hallett  , Mark                   : How physiology speaks to the issue of responsibility", \
                                "Prendinger   , Helmut                  , Christian; Ishizuka, Mitsuru; 	A STUDY IN USERS'PHYSIOLOGICAL RESPONSE TO AN EMPATHIC INTERFACE AGENT", \
                                "Brown    , Ronald              Delp, Michael D; Lindstedt, Stan L; Rhomberg, Lorenz R; Beliles, Robert P; 	Physiological parameter values for physiologically based pharmacokinetic models", \
                                "Cochrane     , Vincent             	Physiology of fungi.", \
                                "Carpenter    , William                       of mental physiology", \
                                "Kauffman     , Stuart                   , Philip; 	On emergence, agency, and organization", \
                                "Okasha   , Samir                  concept of agent in biology: motivations and meanings"]


# Queries
# used general: 
# queries = ["sense of agency", "agency", "theory of agency"]
num_results = 1
data = get_publication_data(queries_from_citations_txt, num_results)
create_csv(data, 'citation_data.csv')

 #                               "Wilson   , Robert              	Genes and the agents of life: The individual in the fragile sciences biology", \
 #                               "Harari   , Paul            Allen, Gregory W; Bonner, James A; 	Biology of interactions: antiepidermal growth factor receptor agents", \
 #                               "Lebowitz     , Matthew             Ahn, Woo-kyoung; 	Emphasizing malleability in the biology of depression: Durable effects on perceived agency and prognostic pessimism", \
 #                               "Mandava  , N                     	Chemistry and biology of allelopathic agents", \
 #                               "Haggard  , Patrick                  , Valerian; 	Sense of agency", \
#                                "Caspar   , Emilie              Christensen, Julia F; Cleeremans, Axel; Haggard, Patrick; 	Coercion changes the sense of agency in the human brain", \
 #                               "Welch    , John            	What’s wrong with evolutionary biology?", \
 #                               "Birke    , Lynda                     and biology", \
 #                               "Yoshie   , Michiko                  , Patrick; 	Negative emotional outcomes attenuate sense of agency over voluntary actions", \
 #                               "Lindner  , Axel               , Peter; Kircher, Tilo TJ; Haarmeier, Thomas; Leube, Dirk T; 	Disorders of agency in schizophrenia correlate with an inability to compensate for the sensory consequences of actions", \
 #                               "Haggard  , Patrick                  of agency in the human brain", \
 #                               "Gallagher    , Shaun                       aspects in the sense of agency", \
 #                               "De    Vignemont,                        Fourneret, Pierre; 	The sense of agency: A philosophical and empirical review of the “Who” system", \
 #                               "Pacherie     , Elisabeth                  sense of control and the sense of agency", \
 #                               "Moore    , James               	What is the sense of agency and why does it matter?", \
 #                               "Moore    , James               Obhi, Sukhvinder S; 	Intentional binding and the sense of agency: a review", \
 #                               "Haggard  , Patrick                , Baruch; 	The sense of agency", \
 #                               "Bayne    , Tim                sense of agency", \
 #                               "Barlas   , Zeynep                , Sukhvinder S; 	Freedom, choice, and the sense of agency", \
 #                               "Jeunet   , Camille                 , Louis; Argelaguet, Ferran; Lécuyer, Anatole; 	“Do you feel in control?”: towards novel approaches to characterise, manipulate and measure the sense of agency in virtual environments", \
 #                               "Chambon  , Valérian                 , Nura; Haggard, Patrick; 	From action intentions to action effects: how does the sense of agency come about?", \
 #                               "Moore    , James               Middleton, D; Haggard, Patrick; Fletcher, Paul C; 	Exploring implicit and explicit aspects of sense of agency", \
 #                               "Moore    , James               Wegner, Daniel M; Haggard, Patrick; 	Modulating the sense of agency with external cues", \
 #                               "Hohwy    , Jakob                  sense of self in the phenomenology of agency and perception", \
 #                               "Engbert  , Kai                       , Andreas; Haggard, Patrick; 	Who is causing what? The sense of agency is relational and efferent-triggered", \
 #                               "Proust   , Joëlle                there a sense of agency for thought", \
 #                               "Farrer   , Chlöé                     , Guilhem; Hupé, Jean-Michel; 	The time windows of the sense of agency", \
 #                               "Moretto  , Giovanna               , Eamonn; Haggard, Patrick; 	Experience of agency and sense of responsibility", \
 #                               "Chambon  , Valerian               , Dorit; Fleming, Stephen M; Prinz, Wolfgang; Haggard, Patrick; 	An online neural substrate for sense of agency", \
 #                               "Longo    , Matthew             Haggard, Patrick; 	Sense of agency primes manual motor responses", \
 #                               "Kush     , Ken                  , Larry; 	Enhancing a sense of agency through career planning.", \
 #                               "Tapal    , Adam              , Ela; Dar, Reuven; Eitam, Baruch; 	The sense of agency scale: A measure of consciously perceived control over one's mind, body, and the immediate environment", \
 #                               "Jeannerod    , Marc               sense of agency and its disturbances in schizophrenia: a reappraisal", \
 #                               "Braun    , Niclas                   , Stefan; Spychala, Nadine; Bongartz, Edith; Sörös, Peter; Müller, Helge HO; Philipsen, Alexandra; 	The senses of agency and ownership: a review", \
 #                               "Buhrmann     , Thomas               Paolo, Ezequiel; 	The sense of agency–a phenomenological consequence of enacting sensorimotor schemes", \
 #                               "Gallagher    , Shaun                        in the sense of agency", \
 #                               "Beck     , Brianna              Costa, Steven; Haggard, Patrick; 	Having control over the external world increases the implicit sense of agency", \
 #                               "Hommel   , Bernhard                  control and the sense of agency", \
 #                               "Obhi     , Sukhvinder              Hall, Preston; 	Sense of agency and intentional binding in joint action", \
 #                               "Dewey    , John            Knoblich, Günther; 	Do implicit and explicit measures of the sense of agency measure the same thing?", \
 #                               "Wen  , Wen                    , Atsushi; Asama, Hajime; 	The sense of agency during continuous action: performance is more important than action-feedback association", \
 #                               "Obhi     , Sukhvinder              Hall, Preston; 	Sense of agency in joint action: Influence of human and computer co-actors", \
 #                               "Hacker   , Douglas             Dunlosky, John; Graesser, Arthur C; 	A growing sense of “agency”", \
 #                               "Wen  , Wen                 , Yoshihiro; Asama, Hajime; 	The sense of agency in driving automation", \
 #                               "David    , Nicole                 , Albert; Vogeley, Kai; 	The “sense of agency” and its underlying cognitive and neural mechanisms", \
 #                               "Wen  , Wen                 delay in feedback diminish sense of agency? A review", \
 #                               "Saito    , Naho                  , Keisuke; Murai, Toshiya; Takahashi, Hidehiko; 	Discrepancy between explicit judgement of agency and implicit feeling of agency: Implications for sense of agency and its disorders", \
 #                               "Ataria   , Yochai                   of ownership and sense of agency during trauma", \
 #                               "Gallese  , Vittorio               inner sense of action. Agency and motor representations", \
 #                               "Imaizumi     , Shu                , Yoshihiko; 	Intentional binding coincides with explicit sense of agency", \
 #                               "Demanet  , Jelle                  -Karbe, Paul S; Lynn, Margaret T; Blotenberg, Iris; Brass, Marcel; 	Power to the will: How exerting physical effort boosts the sense of agency", \
 #                               "Borhani  , Khatereh              , Brianna; Haggard, Patrick; 	Choosing, doing, and controlling: implicit sense of agency over somatosensory events", \
 #                               "Feldman  , Gilad                     sense of agency: Belief in free will as a unique and important construct", \
 #                               "Berberian    , Bruno                     , Jean-Christophe; Le Blaye, Patrick; Haggard, Patrick; 	Automation technology and sense of control: a window on human agency", \
 #                               "Leslie   , Alan            	A theory of agency", \
 #                               "David    , Nicole                 frontiers in the neuroscience of the sense of agency", \
 #                               "Obhi     , Sukhvinder              Swiderski, Kristina M; Brubacher, Sonja P; 	Induced power changes the sense of agency", \
 #                               "Cornelio     , Patricia                 , Patrick; Hornbaek, Kasper; Georgiou, Orestis; Bergström, Joanna; Subramanian, Sriram; Obrist, Marianna; 	The sense of agency in emerging technologies for human–computer integration: A review", \
 #                               "Li   , Jin                 in learning: Chinese adolescents' goals and sense of agency", \
 #                               "van   der Wel                     PRD; Sebanz, Natalie; Knoblich, Guenther; 	The sense of agency during skill learning in individuals and dyads", \
 #                               "Lukoff   , Kai                , Ulrik; Zade, Himanshu; Liao, J Vera; Choi, James; Fan, Kaiyue; Munson, Sean A; Hiniker, Alexis; 	How the design of youtube influences user sense of agency", \
 #                               "Sidarus  , Nura                , Matti; Haggard, Patrick; 	How action selection influences the sense of agency: An ERP study", \
 #                               "Haering  , Carola                  , Andrea; 	Was it me when it happened too early? Experience of delayed effects shapes sense of agency", \
 #                               "Polito   , Vince                    , Amanda J; Woody, Erik Z; 	Developing the Sense of Agency Rating Scale (SOARS): An empirical measure of agency disruption in hypnosis", \
 #                               "Zito     , Giuseppe            Wiest, Roland; Aybek, Selma; 	Neural correlates of sense of agency in motor control: A neuroimaging meta-analysis", \
 #                               "Hon  , Nicholas             , Jia-Hou; Soon, Chun-Siong; 	Preoccupied minds feel less control: Sense of agency is modulated by cognitive load", \
 #                               "Moore    , James               Ruge, Diane; Wenke, Dorit; Rothwell, John; Haggard, Patrick; 	Disrupting the experience of control in the human brain: pre-supplementary motor area contributes to the sense of agency", \
 #                               "Legaspi  , Roberto                    , Taro; 	A Bayesian psychophysics model of sense of agency", \
 #                               "Minohara     , Rin              , Wen; Hamasaki, Shunsuke; Maeda, Takaki; Kato, Motoichiro; Yamakawa, Hiroshi; Yamashita, Atsushi; Asama, Hajime; 	Strength of intentional effort enhances the sense of agency", \
 #                               "Haggard  , Patrick                   , Manos; 	The experience of agency: Feelings, judgments, and responsibility", \
 #                               "O    'Meara,                      Campbell, Corbin M; 	Faculty sense of agency in decisions about work and family", \
 #                               "Holma    , Juha                  , Jukka; 	The sense of agency and the search for a narrative in acute psychosis", \
 #                               "Wen  , Wen                    , Atsushi; Asama, Hajime; 	The influence of action-outcome delay and arousal on sense of agency and the intentional binding effect", \
 #                               "Kawabe   , Takahiro                  , Warrick; Nishida, Shin'ya; 	The sense of agency is action–effect causality perception based on cross-modal grouping", \
 #                               "Ahearn   , Laura               	Language and agency", \
 #                               "Holmes   , Oliver                    	Agency. II", \
 #                               "Maxim    , Hiram                  sixth sense of the bat", \
 #                               "Seavey   , Warren              	The Rationale of Agency", \
 #                               "Story    , Joseph                     , Charles Pelham; 	Commentaries on the Law of Agency as a Branch of Commercial and Maritime Jurisprudence, with Occasional Illustrations from the Civil and Foreign Law", \
 #                               "Howe     , Frederic            	The city as a socializing agency: The physical basis of the city: The city plan", \
 #                               "Cook     , Walter                    	Agency by Estoppel", \
 #                               "James    , William                experience of activity.", \
 #                               "Allen    , Grant                  colour-sense: Its origin and development. An essay in comparative psychology", \
 #                               "Wells    , William                   	The Life and Public Services of Samuel Adams: Being a Narrative of His Acts and Opinions, and of His Agency in Producing and Forwarding the American Revolution. With Extracts from His Correspondence, State Papers, and Political Essays", \
 #                               "Vinogradoff  , Paul                 -sense in Law", \
 #                               "Synofzik     , Matthis                   , Gottfried; Newen, Albert; 	Beyond the comparator model: a multifactorial two-step account of agency", \
 #                               "Carruthers   , Glenn                comparison of fortunes: the comparator and multifactorial weighting models of the sense of agency", \
 #                               "Carruthers   , Glenn                metacognitive model of the feeling of agency over bodily actions.", \
 #                               "Lukitsch     , Oliver                   , uncertainty, and the sense of agency", \
 #                               "Steward  , Helen                     agency", \
 #                               "Špinka   , Marek                     agency, animal awareness and animal welfare", \
 #                               "Špinka   , Marek                         , Françoise; 	Environmental challenge and animal agency", \
 #                               "Blattner     , Charlotte               Donaldson, Sue; Wilcox, Ryan; 	Animal agency in community", \
 #                               "Carter   , Bob                  , Nickie; 	Animals, agency and resistance", \
 #                               "Edelblutte   , Émilie                       , Roopa; Hayek, Matthew Nassif; 	Animal agency in wildlife conservation and management", \
 #                               "Mc   Farland,                  Hediger, Ryan; 	Animals and agency: an interdisciplinary exploration", \
 #                               "Jamieson     , Dale                  agency", \
 #                               "Lindstrøm    , Torill                      	Agency ‘in itself’. A discussion of inanimate, animal and human agency", \
 #                               "Mc   Farland,                  ; Hediger, Ryan; 	Approaching the agency of other animals: An introduction", \
 #                               "Špinka   , M                         , Françoise; 	Environmental challenge and animal agency.", \
 #                               "Howell   , Philip                    , agency, and history", \
 #                               "Shapiro  , Paul                 agency in other animals", \
 #                               "Glock    , Hans                 	Animal agency", \
 #                               "Glock    , Hans                 	Agency, intelligence and reasons in animals", \
 #                               "Shaw     , David                  	The torturer's horse: Agency and animals in history", \
 #                               "Bull     , Jacob                     Movements-Moving Animals: essays on direction, velocity and agency in humanimal encounters", \
 #                               "Radick   , Gregory                   agency in the age of the Modern Synthesis: WH Thorpe's example", \
 #                               "Mc   Hugh,                 	Literary animal agents", \
 #                               "Wilcox   , Marc            	Animals and the agency account of moral status", \
 #                               "Pepper   , Angie                        agency in humans and other animals", \
 #                               "Nance    , Susan                           elephants: Animal agency and the business of the American circus", \
 #                               "Colditz  , Ian             	Objecthood, agency and mutualism in valenced farm animal environments", \
 #                               "Sueur    , Cédric                 , Sarah; Pelé, Marie; 	Incorporating animal agency into research design could improve behavioral and neuroscience research.", \
 #                               "Delon    , Nicolas                   agency, captivity, and meaning", \
 #                               "Newman   , Stuart              Corning, PA; Kauffman, SA; Noble, D; Shapiro, JA; Vane-Wright, RI; Pross, A; 	Form, function, agency: sources of natural purpose in animal evolution", \
 #                               "Samy     , Ramar                     Stiles, Bradley G; Franco, Octavio L; Sethi, Gautam; Lim, Lina HK; 	Animal venoms as antimicrobial agents", \
 #                               "Blattner     , Charlotte               	Turning to animal agency in the Anthropocene", \
 #                               "Barandiaran  , Xabier              	Autonomy and enactivism: towards a theory of sensorimotor autonomous agency", \
 #                               "Gallagher    , Shaun                  , Matthew; 	Making enactivism even more embodied", \
 #                               "Maiese   , Michelle                      , enactivism, and ideological oppression", \
 #                               "Ward     , Dave                   , David; Villalobos, Mario; 	Introduction: The varieties of enactivism", \
 #                               "Heras    -Escribano,                    Noble, Jason; De Pinedo, Manuel; 	Enactivism, action and normativity: a Wittgensteinian analysis", \
 #                               "Maiese   , Michelle                   , enactivism, and mental disorder: A philosophical account", \
 #                               "De    Jesus,                   	Thinking through enactive agency: sense-making, bio-semiosis and the ontologies of organismic worlds", \
 #                               "Van   Grunsven,                	Enactivism, second-person engagement and personal responsibility", \
 #                               "Degenaar     , Jan                  , J Kevin; 	Sensorimotor theory and enactivism", \
 #                               "Ward     , Dave                     transparency: An argument for enactivism", \
 #                               "Merritt  , Michele                    -is-moving: dance, agency, and a radically enactive mind", \
 #                               "Read     , Catherine                       , Agnes; 	Ecological psychology and enactivism: perceptually-guided action vs. sensation-based enaction", \
 #                               "Kovács   , Dániel                   	Plotinus on the Parthood and Agency of Individual Souls", \
 #                               "Polkowski    , Lech                     in engineering and computer science", \
 #                               "Johnson  , Julia                    mereology for industrial design", \
 #                               "Rose     , David                     , Jonathan; 	Folk mereology is teleological", \
 #                               "Carrara  , Massimiliano                 , Enrico; 	On the ontological commitment of mereology", \
 #                               "De    Vignemont,                        Tsakiris, Manos; Haggard, Patrick; 	Body mereology", \
 #                               "Polkowski    , Lech              connection synthesis via rough mereology", \
 #                               "Polkowski    , Lech                 , Andrzej; 	Approximate reasoning about complex objects in distributed systems: Rough mereological formalization", \
 #                               "Zmudzinski   , Lukasz                     , Piotr; 	Path planning based on potential fields from rough mereology", \
 #                               "Polkowski    , Lech                 , Andrzej; 	Introducing rough mereological controllers: Rough quality control", \
 #                               "Giberman     , Daniel                Mereology a Guide to Conceivability?", \
 #                               "Florio   , Salvatore                    , David; 	Plurals and mereology", \
 #                               "Mahmood  , Mahmood             El-Bendary, Nashwa; Hassanien, Aboul Ella; Hefny, Hesham A; 	An Intelligent Multi-Agent Recommender System using Rough Mereology", \
 #                               "Polkowski    , Lech                 , Andrzej; Komorowski, Jan; 	Approximate case-based reasoning: A rough mereological approach", \
 #                               "Klinov   , Pavel                   , Julia M; Mazlack, Lawrence J; 	Interval rough mereology and description logic: An approach to formal treatment of imprecision in the Semantic Web ontologies"]#