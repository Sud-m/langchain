from ast import TemplateStr
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def main():
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman, industrialist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, Forbes estimates his net worth to be $1.1 trillion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service, and made significant changes, including rebranding it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion. Musk became the first US-dollar trillionaire upon the initial public offering of SpaceX in June 2026, though a SpaceX stock collapse brought him under the threshold by the next month.

Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated in January 2025, Musk was an advisor to Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk left the Trump administration in May 2025 and returned to managing his companies; shortly thereafter he had a public feud with Trump.

Musk is a supporter of global far-right politics, figures, and political parties. His political activities, statements and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including spreading COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, white nationalist, racist, and transphobic comments. His acquisition of Twitter was controversial because, following his pledge to decrease censorship, there was an increase in hate speech and misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE and its cuts to the US Agency for International Development (USAID).

Early life and education
See also: Musk family
Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital.[3][4] He is of British and Pennsylvania Dutch ancestry.[5][6] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[7][8][9] Musk therefore holds both South African and Canadian citizenship from birth.[10] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve.[11][12][13][14]

His maternal grandfather, Joshua N. Haldeman, who died in a plane crash when Musk was a toddler, was an American-born Canadian chiropractor, aviator and political activist in the technocracy movement[15][16] who moved to South Africa in 1950.[17] Haldeman's anti-government, anti-democratic and conspiracist views, which included the promotion of far-right antisemitic conspiracy theories,[18][19] "fanatical" support of apartheid,[19] and according to Errol, support of Nazism,[17] have been suggested as an influence on Musk.[20][21][22][23] During his childhood, Musk was told stories by his grandmother of Haldeman's travels and exploits, and he has suggested that all of Haldeman's descendants have his "desire for adventure, exploration – doing crazy things".[24]

Musk has a younger brother, Kimbal, a younger sister, Tosca, and four paternal half-siblings.[25][26][9][27] He was baptized as a child in the Anglican Church of Southern Africa.[28][29] The Musk family was wealthy during his youth.[14] Despite both Musk and Errol previously stating that Errol was a part owner of a Zambian emerald mine,[14] in 2023, Errol recounted that the deal he made was to receive "a portion of the emeralds produced at three small mines".[30][31] Errol was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.[3]

After his parents divorced in 1979, Musk, aged around 9, chose to live with his father because he had an Encyclopædia Britannica set and a computer.[32][5][11] He later regretted his decision and became estranged from his father.[33] Musk has recounted trips to a wilderness school that he described as a "paramilitary Lord of the Flies" where "bullying was a virtue" and children were encouraged to fight over rations.[34] In one incident, after an altercation with a fellow pupil, Musk was thrown down concrete steps and beaten severely, leading to him being hospitalized for his injuries.[35] He described his father berating him after he was discharged from the hospital.[35] Errol denied berating Musk and claimed, "The [other] boy had just lost his father to suicide, and Musk had called him stupid. He had a tendency to call people stupid. How could I possibly blame that child?"[36]

Musk was an enthusiastic reader of books, and had attributed his success in part to having read The Lord of the Rings, the Foundation series, and The Hitchhiker's Guide to the Galaxy.[13][37] At age ten, he developed an interest in computing and video games, teaching himself how to program from the VIC-20 user manual.[38] At age twelve, in 1983, Musk sold his BASIC-based game Blastar to PC and Office Technology magazine for approximately $500 (equivalent to $1,600 in 2025).[39][40]

Education
An ornate school building
Musk graduated from Pretoria Boys High School in Pretoria, South Africa.
Musk attended Waterkloof House Preparatory School, Bryanston High School, and then Pretoria Boys High School, where he graduated.[41] He was a decent but unexceptional student, earning a 61/100 in Afrikaans and a B on his senior math certification.[42] He applied for a Canadian passport through his Canadian-born mother to avoid South Africa's mandatory military service,[43][44] which would have forced him to participate in the apartheid regime's mandatory military service for young white South African men,[3] as well as to ease his path to immigration to the United States.[45] While waiting for his application to be processed, he attended the University of Pretoria for five months.[46]

Musk arrived in Canada in June 1989, connected with a second cousin in Saskatchewan,[47][48] and worked odd jobs, including at a farm and a lumber mill.[49] In 1990, he entered Queen's University in Kingston, Ontario.[50][51] Two years later, he transferred to the University of Pennsylvania, where he studied until 1995.[52] Although Musk has said that he earned his degrees in 1995, the University of Pennsylvania did not award them until 1997 – a Bachelor of Arts in physics and a Bachelor of Science in economics from the university's Wharton School.[53][54][55][56][57] He reportedly hosted large, ticketed house parties to help pay for tuition, and wrote a business plan for an electronic book-scanning service similar to Google Books.[58]

In 1994, Musk held two internships in Silicon Valley: one at energy storage startup Pinnacle Research Institute, which investigated electrolytic supercapacitors for energy storage, and another at Palo Alto–based startup Rocket Science Games.[59][60] In 1995, he was accepted to a graduate program in materials science at Stanford University, but did not enroll.[55][53][61] Musk decided to join the Internet boom of the 1990s, applying for a job at Netscape, to which he reportedly never received a response.[62][43] The Washington Post reported that Musk lacked legal authorization to remain and work in the United States after failing to enroll at Stanford.[61] In response, Musk said he was allowed to work at that time and that his student visa transitioned to an H1-B. According to numerous former business associates and shareholders, Musk said he was on a student visa at the time.[63]

Business career
Main article: Business career of Elon Musk
Zip2
Main article: Zip2
External videos
video icon Musk speaks of his early business experience during a 2014 commencement speech at University of Southern California on YouTube
In 1995, Musk, his brother Kimbal, and Greg Kouri founded Zip2, a web software company, with funding from a group of angel investors.[64] They housed the venture at a small rented office in Palo Alto.[65] Replying to Rolling Stone, Musk denounced the notion that they started their company with funds borrowed from Musk's father Errol Musk,[33] but in a tweet, he recognized that his father contributed 10% of a later funding round.[66] The company developed and marketed an Internet city guide for the newspaper publishing industry, with maps, directions, and yellow pages.[67]

According to Musk, "The website was up during the day and I was coding it at night, seven days a week, all the time."[65] To impress investors, Musk built a large plastic structure around a standard computer to create the impression that Zip2 was powered by a small supercomputer.[68] The Musk brothers obtained contracts with The New York Times and the Chicago Tribune,[69] and persuaded the board of directors to abandon plans for a merger with CitySearch.[70] Musk's attempts to become CEO were thwarted by the board.[71] Compaq acquired Zip2 for $307 million in cash in February 1999 (equivalent to $590,000,000 in 2025),[72][73] and Musk received $22 million (equivalent to $43,000,000 in 2025) for his 7-percent share.[74]

X.com and PayPal
Main articles: X.com (bank), PayPal, and PayPal Mafia
In 1999, Musk co-founded X.com, an online financial services and e-mail payment company.[75] The startup was one of the first federally insured online banks, and, in its initial months of operation, over 200,000 customers joined the service.[76] The company's investors regarded Musk as inexperienced and replaced him with Intuit CEO Bill Harris by the end of the year.[77] The following year, X.com merged with online bank Confinity to avoid competition.[65][77][78] Founded by Max Levchin and Peter Thiel,[79] Confinity had its own money-transfer service, PayPal, which was more popular than X.com's service.[80]

Within the merged company, Musk returned as CEO. Musk's preference for Microsoft software over Unix created a rift in the company and caused Thiel to resign.[81] Due to resulting technological issues and lack of a cohesive business model, the board ousted Musk and replaced him with Thiel in 2000.[82][d] Under Thiel, the company focused on the PayPal service and was renamed PayPal in 2001.[84][85] In 2002, PayPal was acquired by eBay for $1.5 billion (equivalent to $2.7×109 in 2025) in stock, of which Musk—the largest shareholder with 11.72% of shares—received $175.8 million (equivalent to $320,000,000 in 2025).[86][87] In 2017, Musk purchased X.com from PayPal for an undisclosed amount, stating that it had sentimental value.[88][89] He later used the domain in his rebrand of Twitter to X.

SpaceX
Main article: SpaceX

Musk explaining Starship capabilities to leaders of North American Aerospace Defense Command, U.S. Northern Command, and Air Force Space Command in 2019
In 2001, Musk became involved with the nonprofit Mars Society and discussed funding plans to place a growth-chamber for plants on Mars.[90] Seeking a way to launch the greenhouse payloads into space, Musk made two unsuccessful trips to Moscow to purchase intercontinental ballistic missiles (ICBMs) from Russian companies NPO Lavochkin and Kosmotras. Musk instead decided to start a company to build affordable rockets.[91] With $100 million of his early fortune,[92] (equivalent to $180,000,000 in 2025) Musk founded SpaceX in May 2002 and became the company's CEO and Chief Engineer.[93][94]

SpaceX attempted its first launch of the Falcon 1 rocket in 2006.[95] Although the rocket failed to reach Earth orbit, it was awarded a Commercial Orbital Transportation Services program contract from NASA, then led by Mike Griffin.[96][97] After two more failed attempts that nearly caused Musk to go bankrupt,[95] SpaceX succeeded in launching the Falcon 1 into orbit in 2008.[98] Later that year, SpaceX received a $1.6 billion NASA contract (equivalent to $2.4×109 in 2025) for Falcon 9-launched Dragon spacecraft flights to the International Space Station (ISS), replacing the Space Shuttle after its 2011 retirement.[99] In 2012, the Dragon vehicle docked with the ISS, a first for a commercial spacecraft.[100]

Working towards its goal of reusable rockets, in 2015 SpaceX successfully landed the first stage of a Falcon 9 on a land platform.[101] Later landings were achieved on autonomous spaceport drone ships, an ocean-based recovery platform.[102] In 2018, SpaceX launched the Falcon Heavy; the inaugural mission carried Musk's personal Tesla Roadster as a dummy payload.[103][104] Since 2019,[105] SpaceX has been developing Starship, a reusable, super heavy-lift launch vehicle intended to replace the Falcon 9 and Falcon Heavy.[106] In 2020, SpaceX launched its first crewed flight, the Demo-2, becoming the first private company to place astronauts into orbit and dock a crewed spacecraft with the ISS.[107] In 2024, NASA awarded SpaceX an $843 million (equivalent to $865,000,000 in 2025) contract to build a spacecraft that NASA will use to deorbit the ISS at the end of its lifespan.[108]

Starlink
Main article: Starlink
See also: Starlink in the Russo-Ukrainian War

50 Starlink satellites shortly before deployment to low Earth orbit, 2019
In 2015, SpaceX began development of the Starlink constellation of low Earth orbit satellites to provide satellite Internet access.[109] After the launch of prototype satellites in 2018, the first large constellation was deployed in May 2019.[110] As of May 2025, over 7,600 Starlink satellites are operational,[111] comprising 65% of all operational Earth satellites.[112] The total cost of the decade-long project to design, build, and deploy the constellation was estimated by SpaceX in 2020 to be $10 billion (equivalent to $1.2×1010 in 2025).[113][e]

During the Russian invasion of Ukraine, Musk provided free Starlink service to Ukraine, permitting Internet access and communication at a yearly cost to SpaceX of $400 million (equivalent to $440,000,000 in 2025).[116][117][118][119][120] However, Musk refused to block Russian state media on Starlink.[121][122] In 2023, Musk denied Ukraine's request to activate Starlink over Crimea to aid an attack against the Russian navy, citing fears of a nuclear response.[123][124][125]

Tesla
Main article: Tesla, Inc.
Musks stands, arms crossed and grinning, before a Tesla Model S
Musk next to a Tesla Model S, 2011
Tesla, Inc., originally Tesla Motors, was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning. Both men played active roles in the company's early development prior to Musk's involvement.[126] Musk led the Series A round of investment in February 2004; he invested $6.35 million (equivalent to $11,000,000 in 2025), became the majority shareholder, and joined Tesla's board of directors as chairman.[127][128] Musk took an active role within the company and oversaw Roadster product design, but was not deeply involved in day-to-day business operations.[129] Following a series of escalating conflicts in 2007 and the 2008 financial crisis, Eberhard was ousted from the firm.[130][page needed][131] Musk assumed leadership of the company as CEO and product architect in 2008.[132] A 2009 lawsuit settlement with Eberhard designated Musk as a Tesla co-founder, along with Tarpenning and two others.[133][134]

Tesla began delivery of the Roadster, an electric sports car, in 2008. With sales of about 2,500 vehicles, it was the first mass production all-electric car to use lithium-ion battery cells.[135] Under Musk, Tesla has since launched several well-selling electric vehicles, including the four-door sedan Model S (2012),[136] the crossover Model X (2015),[137] the mass-market sedan Model 3 (2017),[138] the crossover Model Y (2020),[139] and the pickup truck Cybertruck (2023).[140] The company has also constructed multiple lithium-ion battery and electric vehicle factories, called Gigafactories.[141]

In May 2020, Musk resigned as chairman of the board as part of the settlement of a lawsuit from the SEC over him tweeting that funding had been "secured" for potentially taking Tesla private.[142][143]

Since its initial public offering in 2010,[144] Tesla stock has risen significantly; it became the most valuable carmaker in summer 2020,[145][146] and it entered the S&P 500 later that year.[147][148] In October 2021, it reached a market capitalization of $1 trillion (equivalent to $1.2×1012 in 2025), the sixth company in U.S. history to do so.[149]

SolarCity and Tesla Energy
Main articles: SolarCity and Tesla Energy
Musk provided the initial concept and financial capital for SolarCity, which his cousins Lyndon and Peter Rive founded in 2006.[150] By 2013, SolarCity was the second largest provider of solar power systems in the United States.[151] In 2014, Musk promoted the idea of SolarCity building an advanced production facility in Buffalo, New York, triple the size of the largest solar plant in the United States.[152] Construction of the factory started in 2014 and was completed in 2017. It operated as a joint venture with Panasonic until early 2020.[153][154]

Tesla acquired SolarCity for $2 billion in 2016 (equivalent to $2.7×109 in 2025) and merged it with its battery unit to create Tesla Energy. The deal's announcement resulted in a more than 10% drop in Tesla's stock price; at the time, SolarCity was facing liquidity issues.[155] Multiple shareholder groups filed a lawsuit against Musk and Tesla's directors, stating that the purchase of SolarCity was done solely to benefit Musk and came at the expense of Tesla and its shareholders.[156][157] Tesla directors settled the lawsuit in January 2020, leaving Musk the sole remaining defendant.[158][159] Two years later, the court ruled in Musk's favor.[155]

Neuralink
Main article: Neuralink
Musk standing next to bulky medical equipment on a stage
Musk discussing a Neuralink device during a live demonstration in 2020
In 2016, Musk co-founded Neuralink, a neurotechnology startup, with an investment of $100 million.[160][161] Neuralink aims to integrate the human brain with artificial intelligence (AI) by creating devices that are embedded in the brain. Such technology could enhance memory or allow the devices to communicate with software.[161][162] The company also hopes to develop devices to treat neurological conditions like spinal cord injuries.[163] In 2022, Neuralink announced that clinical trials would begin by the end of the year.[164] In September 2023, the Food and Drug Administration approved Neuralink to initiate six-year human trials.[165]

Neuralink has conducted animal testing on macaques at the University of California, Davis. In 2021, the company released a video in which a macaque played the video game Pong via a Neuralink implant. The company's animal trials—which have caused the deaths of some monkeys—have led to claims of animal cruelty. The Physicians Committee for Responsible Medicine has alleged that Neuralink violated the Animal Welfare Act.[166] Employees have complained that pressure from Musk to accelerate development has led to botched experiments and unnecessary animal deaths. In 2022, a federal probe was launched into possible animal welfare violations by Neuralink.[167][needs update]

The Boring Company
Main article: The Boring Company
Musk speaks to a crowd of journalists. Behind him is a lighted tunnel.
Musk during the 2018 inauguration of the Boring test tunnel in Hawthorne, California
In 2017, Musk founded the Boring Company to construct tunnels; he also revealed plans for specialized, underground, high-occupancy vehicles that could travel up to 150 miles per hour (240 km/h) and thus circumvent above-ground traffic in major cities.[168][169] Early in 2017, the company began discussions with regulatory bodies and initiated construction of a 30-foot (9.1 m) wide, 50-foot (15 m) long, and 15-foot (4.6 m) deep "test trench" on the premises of SpaceX's offices, as that required no permits.[170]

The Los Angeles tunnel, less than two miles (3.2 km) in length, debuted to journalists in 2018. It used Tesla Model Xs and was reported to be a rough ride while traveling at suboptimal speeds.[171] Two tunnel projects announced in 2018, in Chicago and West Los Angeles, have been canceled.[172][173] A tunnel beneath the Las Vegas Convention Center was completed in early 2021.[174] Local officials have approved further expansions of the tunnel system.[175]

X Corp.
Main articles: Acquisition of Twitter by Elon Musk and Twitter under Elon Musk
Avatar of Elon Musk
Elon Musk 
(@elonmusk)
tweeted:
I made an offer
https://sec.gov/Archives/edgar...
April 14, 2022[176]

In early 2017, Musk expressed interest in buying Twitter and had questioned the platform's commitment to freedom of speech.[177][178][179] By 2022, Musk had reached 9.2% stake in the company,[180] making him the largest shareholder.[181][f] Musk later agreed to a deal that would appoint him to Twitter's board of directors and prohibit him from acquiring more than 14.9% of the company.[183][184] Days later, Musk made a $43 billion offer to buy Twitter.[181][185] By the end of April Musk had successfully concluded his bid for approximately $44 billion.[186] This included approximately $12.5 billion in loans and $21 billion in equity financing.[187][188] Having backtracked on his initial decision,[189] Musk bought the company on October 27, 2022.[190]

Immediately after the acquisition, Musk fired several top Twitter executives including CEO Parag Agrawal;[190][191] Musk became the CEO instead.[192] Under Musk, Twitter instituted monthly subscriptions for a "blue check",[193][194][195] and laid off a significant portion of the company's staff.[196][197] Musk lessened content moderation and hate speech also increased on the platform after his takeover.[198][199][200][201] In late 2022, Musk released internal documents relating to Twitter's moderation of Hunter Biden's laptop controversy in the lead-up to the 2020 presidential election.[202] Musk also promised to step down as CEO after a Twitter poll,[203][204] and five months later, Musk stepped down as CEO and transitioned his role to executive chairman and chief technology officer (CTO).[205]

Despite Musk stepping down as CEO, X continues to struggle with challenges such as viral misinformation,[206] hate speech, and antisemitism controversies.[207][208] Musk has been accused of trying to silence some of his critics such as Twitch streamer Asmongold, who criticized him during one of his streams.[209] Musk has been accused of removing their accounts' blue checkmarks, which hinders visibility and is considered a form of shadow banning,[210][211] or suspending their accounts without justification.[212]

Other activities
Hyperloop
Main articles: Hyperloop and Hyperloop pod competition
A long white tube about 10 feet in diameter
A tube part of the 2017 Hyperloop pod competition, sponsored by SpaceX
In August 2013, Musk announced plans for a version of a vactrain, and assigned engineers from SpaceX and Tesla to design a transport system between Greater Los Angeles and the San Francisco Bay Area, at an estimated cost of $6 billion.[213][214] Later that year, Musk unveiled the concept, dubbed the Hyperloop,[215] intended to make travel cheaper than any other mode of transport for such long distances.[216]

OpenAI and xAI
Main articles: OpenAI and xAI (company)
In December 2015, Musk co-founded OpenAI, a not-for-profit artificial intelligence (AI) research company aiming to develop artificial general intelligence, intended to be safe and beneficial to humanity.[217] Musk pledged $1 billion of funding to the company,[218] but only donated $50 million.[219] In 2018, Musk left the OpenAI board.[220] Since 2018, OpenAI has made significant advances in machine learning.[221] In July 2023, Musk launched the artificial intelligence company xAI, which aims to develop a generative AI program that competes with existing offerings like OpenAI's ChatGPT. Musk obtained funding from investors in SpaceX and Tesla,[222] and xAI hired engineers from Google and OpenAI.[223]

Private jet
Main articles: ElonJet and 2022 Twitter suspensions
Avatar of Elon Musk
Elon Musk
(@elonmusk)
tweeted:
Same doxxing rules apply to "journalists" as to everyone else
December 16, 2022[224]

Musk uses a private jet owned by Falcon Landing LLC, a SpaceX-linked company, and acquired a second jet in August 2020.[225][226] His heavy use of the jets and the consequent fossil fuel usage have received criticism.[225][227] Musk's flight usage is tracked on social media through ElonJet.[228][229][230] In December 2022, Musk banned the ElonJet account on Twitter, and made temporary bans on the accounts of journalists that posted stories regarding the incident, including Donie O'Sullivan, Keith Olbermann, and journalists from The New York Times, The Washington Post, CNN, and The Intercept.[231]

Grok
This section is an excerpt from Grok (chatbot).edit
Grok is a series of generative AI large language models developed by SpaceXAI[232]. It was launched in November 2023 by Elon Musk as an initiative based on the large language model (LLM) of the same name. Grok has apps for iOS and Android and is integrated with the X social network and Tesla's Optimus robot where it has chatbot functionality. Grok is named after the verb to grok, created by the American science fiction author Robert A. Heinlein to convey a form of deep, intuitive understanding.

SpaceXAI has released several Grok models. Grok-1 was open-sourced in 2024,[233] followed by Grok-2, Grok 3, and Grok 4 model families,[234] with Grok 4.5 released in 2026.[235] Different versions added capabilities including image generation,[236] web search,[237] "Think" mode to simulate reasoning,[238], agentic coding tool called Grok Build,[239] and AI teammates called Grok Bot.[240] Musk admitted in 2026 that Grok was partially distilled from OpenAI's GPT models.[241][242] Starting with Grok 4.5, the models are co-developed with incoming SpaceXAI subsidiary Cursor.[235]

The bot has received criticism for promoting conspiracy theories, praising Adolf Hitler, using antisemitic tropes, and creating nonconsensual, sexualized images of undressed women and children.

Grok generates articles for the encyclopedia website Grokipedia, launched in October 2025 by Musk as an alternative to Wikipedia, which he described as "woke" and politically biased.

Grokipedia
Main article: Grokipedia
In October 2025, Musk's company xAI launched Grokipedia, an AI-generated online encyclopedia that he promoted as an alternative to Wikipedia.[243][244] Articles on Grokipedia were generated and reviewed by xAI's Grok chatbot[245] until the site effectively froze in April 2026.[246]

Media coverage and academic analysis described Grokipedia as frequently reusing Wikipedia content but framing contested political and social topics in line with Musk's own views and right-wing narratives. A study by Cornell University researchers and NBC News stated that Grokipedia cites sources that are blacklisted or considered "generally unreliable" on Wikipedia, for example, the conspiracy site Infowars and the neo-Nazi forum Stormfront. Wired, The Guardian and Time criticized Grokipedia for factual errors and for presenting Musk himself in unusually positive terms while downplaying controversies.[243][247][248][249]

Political positions
Main articles: Views of Elon Musk and Political activities of Elon Musk
See also: Protests against Elon Musk
Part of a series on
Conservatism
in the United States

Schools
Principles
History
Intellectuals
Politicians
Jurists
Commentators
Activists
Literature
Parties
Think tanks
Media
Other organizations
Movements
Related
 Conservatism portal
flag United States portal
vte
Part of a series on
American nationalism

Ideologies
Principles
History
Intellectuals
Politicians
Activists and commentators
Parties
Literature
Organizations
Movements
Related
flag United States portal
 Conservatism portal
vte
Part of a series on
Trumpism

Principles
People
TrumpAdamsBannonBoebertBondiBruesewitzBuchananCarlsonChansleyBuskirkEpshteynGaetzGosarGrayGreeneHagemanHawleyHegsethR. Kennedy Jr.KimKirk (Charlie)Kirk (Erika)MillerMuskNoemPaladinoPatelPaxtonRaiklinRamaswamyRatcliffeRobertsShapiroSt. ClairTararTrump Jr.VanceWilesWoodward
Parties
Related
flag United States portal
 Conservatism portal
icon Politics portal
vte

This section may be unbalanced toward certain viewpoints. Please help improve it by adding information on neglected viewpoints. Relevant discussion may be found on the talk page. (April 2026)
Avatar of Elon Musk
Elon Musk
(@elonmusk)
tweeted:
My commitment to free speech extends even to not banning the account following my plane, even though that is a direct personal safety risk
November 6, 2022[250]

Musk is an outlier among business leaders who typically avoid partisan political advocacy.[251][252][253] Rejecting the conservative label, Musk has described himself as a political moderate,[254] even as his views have become more right-wing over time.[255] His views have been characterized as libertarian and far-right,[256][257] and after his involvement in European politics, they have received criticism from world leaders such as Emmanuel Macron and Olaf Scholz.[258][259][260][261]

Within the context of American politics, Musk supported Democratic candidates up until 2022, at which point he voted for a Republican for the first time.[262][263] He has stated support for universal basic income,[264] gun rights,[265] freedom of speech,[266] a tax on carbon emissions,[267] and H-1B visas.[268] Musk has expressed concern about issues such as artificial intelligence (AI)[269] and climate change,[270] and has been a critic of wealth tax,[271] short-selling,[272] and government subsidies.[273] An immigrant himself, Musk has been accused of being anti-immigration, and regularly blames immigration policies for illegal immigration.[274] He is also a pronatalist who believes population decline is the biggest threat to civilization,[275] and identifies as a cultural Christian.[276][277] Musk has long been an advocate for space colonization, especially the colonization of Mars. He has repeatedly pushed for humanity colonizing Mars, in order to become an interplanetary species and lower the risks of human extinction.[278]

Musk has promoted conspiracy theories and made controversial statements that have led to accusations of racism, sexism, antisemitism,[279][280] transphobia,[281] disseminating disinformation, and support of white pride.[282][283] While describing himself as a "pro-Semite",[284] his comments regarding George Soros and Jewish communities have been condemned by the Anti-Defamation League and the Biden White House.[285] Musk was criticized during the COVID-19 pandemic for making unfounded epidemiological claims,[286] defying COVID-19 lockdowns restrictions,[287] and supporting the Canada convoy protest against vaccine mandates.[288][289] He has amplified false claims of white genocide in South Africa.[290]

Musk was a registered independent voter when he lived in California.[291] Historically, he has donated to both Democrats and Republicans,[292] many of whom serve in states in which he has a vested interest.[293] Musk supported Democrat Barack Obama in 2008 and 2012,[262] Democrat Hillary Clinton in 2016, and Democrat Joe Biden in 2020.[294] In the 2020 Democratic Party presidential primaries, Musk endorsed candidate Andrew Yang and expressed support for Yang's proposed universal basic income,[295] and endorsed Kanye West's 2020 presidential campaign.[296]

On August 5, 2021, Biden signed an Executive Order targeting that half of cars sold in 2030 should be electric.[297] General Motors CEO Mary Barra was in attendance.[297] Musk commented, "seems odd that Tesla wasn't invited."[298] Barra said in 2024 Musk never forgave not being invited.[299][300] In 2025, former vice president Kamala Harris said that she believed that not inviting Musk was a "big mistake."[301][302] In September 2021, Musk criticized Biden as "not the friendliest administration."[301] In December 2021, Musk publicly expressed opposition to the Build Back Better Act.[303][304]

Since 2022, his political contributions have mostly supported Republicans, with his first vote for a Republican going to Mayra Flores in the 2022 Texas's 34th congressional district special election.[305][306] In 2022, Musk gave over $50 million to Citizens for Sanity, a conservative political action committee.[307] In 2023, he supported Republican Ron DeSantis for the 2024 U.S. presidential election, giving $10 million to his campaign,[307] and hosted DeSantis's campaign announcement on a Twitter Spaces event.[308][309][310] From June 2023 to January 2024, Musk hosted a bipartisan set of X Spaces with Republican and Democratic candidates, including Robert F. Kennedy Jr.,[311] Vivek Ramaswamy,[312] and Dean Phillips.[313]

In 2024, he started supporting international far-right political parties, activists, and causes,[314] and has shared far-right misinformation[315][316][317] and numerous conspiracy theories.[318][319] Since 2024, his views have been generally described as right-wing.[320]

Avatar of Elon Musk
Elon Musk
(@elonmusk)
tweeted:
I fully endorse President Trump and hope for his rapid recovery
July 13, 2024[321]

By the middle of 2024, Musk had become a vocal and financial supporter of Donald Trump.[322] In July 2024, minutes after Trump survived an assassination attempt while speaking in Butler, Pennsylvania, Musk endorsed him for president, saying: "I fully endorse President Trump and hope for his rapid recovery."[323][324] During the presidential campaign, Musk joined Trump on stage at a campaign rally,[325] and during the campaign promoted conspiracy theories and falsehoods about Democrats, election fraud[326] and immigration, in support of Trump.[327][328] Musk was the largest individual donor of the 2024 election, spending $291 million.[329][330] One PAC funded by Musk created and aired deceptive "false-flag" advertisements appearing to come from progressive organizations in order to misrepresent Kamala Harris's positions.[331][332] Jacob Silverman, author of the 2025 book Gilded Rage: Elon Musk and the Radicalization of Silicon Valley, said that the tech industry represented by Musk, Thiel, Andreessen and other capitalists, actually flourished under Biden, but the tech leaders chose Trump for their common ground on cultural issues.[333]

In 2023, Musk said he shunned the World Economic Forum (WEF) because it was boring. The organization commented that they had not invited him since 2015.[334] Musk made a last-minute WEF debut in an interview with the organization's interim co-chair Larry Fink during the 2026 WEF.[335][336] He also participated in Dialog, dubbed "Tech Bilderberg" and organized by Peter Thiel and Auren Hoffman.[337][338]

Starting in January 2025, Musk's international political actions and comments came under increasing scrutiny and criticism, especially from the governments and leaders of France, Germany, Norway, Spain and the United Kingdom, particularly due to his position in the U.S. government from January to May 2025,[339] as well as ownership of X.[340][341][342] A February 2025 NBC News analysis found he had boosted far-right political movements to cut immigration and curtail regulation of business in at least 18 countries on six continents since 2023.[343]

In 2025, Musk contributed $19 million to the Wisconsin Supreme Court race, hoping to influence the state's future redistricting efforts and its regulations governing car manufacturers and dealers.[344][345] Musk was the largest individual donor in the 2026 United States elections, spending over $366 million in support of Republican candidates, including $158 million to the Senate Leadership Fund, $76 million to America PAC, $20 million to the Congressional Leadership Fund, $10 million to Lone Star Liberty PAC, $10 million to the Republican Governors Association, and $2 million to the Republican Attorneys General Association.[346]

Salute at Trump's second inauguration
Main article: Elon Musk salute controversy

Musk giving a gesture at the second inauguration of Donald Trump before saying "My heart goes out to you. It is thanks to you that the future of civilization is assured."[347][348]
During his speech after the second inauguration of Donald Trump, Musk twice made a gesture interpreted by many as a Nazi or a fascist Roman salute.[g] He thumped his right hand over his heart, fingers spread wide, and then extended his right arm out, emphatically, at an upward angle, palm down and fingers together. He then repeated the gesture to the crowd behind him. As he finished the gestures, he said to the crowd, "My heart goes out to you. It is thanks to you that the future of civilization is assured."[347][348][350]

It was widely condemned as an intentional Nazi salute in Germany,[351][352][353] where making such gestures is illegal.[354] The Anti-Defamation League said it was not a Nazi salute,[355] but other Jewish organizations disagreed and condemned the salute.[356][357][358][359] American public opinion was divided on partisan lines as to whether it was a fascist salute.[360] Musk dismissed the accusations of Nazi sympathies, deriding them as "dirty tricks" and a "tired" attack.[361][362] Neo-Nazi and white supremacist groups celebrated it as a Nazi salute.[355][363] Multiple European political parties demanded that Musk be banned from entering their countries.[364][365]

Department of Government Efficiency
Main article: Department of Government Efficiency

Musk wielding a chainsaw at the Conservative Political Action Conference (CPAC) in 2025, imitating a publicity stunt used by Javier Milei symbolic of efficiency, federal mass layoffs and tax cutting
The concept of DOGE emerged in a discussion between Musk and Donald Trump, and in August 2024, Trump committed to giving Musk an advisory role, with Musk accepting the offer.[366] In November and December 2024, Musk suggested that the organization could help to cut the U.S. federal budget, consolidate the number of federal agencies,[367][368] and eliminate the Consumer Financial Protection Bureau,[369][370] and that its final stage would be "deleting itself".[371]

In January 2025, the organization was created by executive order, and Musk was designated a "special government employee".[372][373] Musk led the organization and was an advisor to Trump,[374] although his official role is not clear.[375] In sworn statement during a lawsuit, the director of the White House Office of Administration stated that Musk "is not an employee of the U.S. DOGE Service or U.S. DOGE Service Temporary Organization", "is not the U.S. DOGE Service administrator", and has "no actual or formal authority to make government decisions himself".[376][377] Trump said two days later that he had put Musk in charge of DOGE.[378] A federal judge has ruled that Musk acted as the de facto leader of DOGE.[379]

Musk's role in the second Trump administration, particularly in response to DOGE, has attracted public backlash. He was criticized for his treatment of federal government employees,[380][381][382] including his influence over the mass layoffs of the federal workforce.[383][384][385] He has prioritized secrecy within the organization[386] and has accused others of violating privacy laws.[372] A Senate report alleged that Musk could avoid up to $2 billion in legal liability as a result of DOGE's actions.[387] In May 2025, Bill Gates accused Musk of "killing the world's poorest children" through his cuts to USAID.[388] Modeling by Boston University estimated the cuts resulted in 400,000 children and 200,000 adults deaths by November.[389][390][391] Musk departed on May 28, 2025, when the special government employee's 130 day deadline expired.[392][393][394]

Feud with Donald Trump
Main article: Trump–Musk feud
Avatar of Elon Musk
Elon Musk
(@elonmusk)
tweeted:
Time to drop the really big bomb:
@realDonaldTrump is in the Epstein files. That is the real reason they have not been made public.

Have a nice day, DJT!
June 5, 2025[395]

After leaving office, Musk criticized the Trump administration's Big Beautiful Bill, calling it a "disgusting abomination" due to its provisions increasing the deficit.[396] A feud began between Musk and Trump, with its most notable event being Musk alleging Trump had ties to sex offender Jeffrey Epstein on X (formerly Twitter) on June 5, 2025.[397][398] Trump responded on Truth Social stating that Musk went "CRAZY" after the "EV Mandate" was purportedly taken away and threatened to cut Musk's government contracts.[399][400] Musk then called for a third Trump impeachment.[401] The next day, Trump stated that he did not wish to reconcile with Musk, and added that Musk would face "very serious consequences" if he funds Democratic candidates.[402] On June 11, Musk publicly apologized for the tweets against Trump before deleting them, saying they "went too far".[403]

International relations
Main article: International relations of Elon Musk

Musk with Israeli president Isaac Herzog in November 2023
Musk has been critical of Israel's actions in the Gaza Strip during the Gaza war,[404] praised China's economic and climate goals,[405][406] suggested that Taiwan and China should resolve cross-strait relations,[407][408] and was described as having a close relationship with the Chinese government.[405][406]

In Europe, Musk expressed support for Ukraine in 2022 during the Russian invasion, recommended referendums and peace deals on the annexed Russia-occupied territories,[409][410] and supported the far-right Alternative for Germany political party in 2024.[411]

Regarding British politics, Musk blamed the 2024 UK riots on mass migration and open borders,[412][413] criticized Prime Minister Keir Starmer for what he described as a "two-tier" policing system,[414][415][413] and was subsequently attacked as being responsible for spreading misinformation and amplifying the far-right.[416] Musk has publicly supported far-right activist Tommy Robinson, and initially pledged financial backing to Reform UK ahead of the next general election. In January 2025, he publicly fell out with Reform UK leader Nigel Farage, calling for Farage to be replaced as party leader.[417][418] Musk subsequently shifted his backing towards Restore Britain, the political party launched by Rupert Lowe.[419]

In February 2026, Musk described Spanish Prime Minister Pedro Sánchez as a "tyrant" following his proposal to prohibit minors under the age of 16 from accessing social media platforms.[420]

Regarding Canadian politics, Musk said in October 2026, two days after the 2026 Quebec general election, that he supports the independence of Quebec, the independence of Alberta, and the independence of Saskatchewan.[421]

Legal affairs
Main article: Legal affairs of Elon Musk
In July 2018, Musk arranged for SpaceX and the Boring Company employees to build a mini-submarine to assist in the rescue of children trapped in a flooded cavern in Thailand. Despite him drawing attention to it on social media,[422] the mini-submarine was deemed not practical for the rescue mission.[423] A British diver interviewed about the mission, Vern Unsworth, reflected on Musk's actions collectively a "PR stunt". In response, Musk referred to Unsworth as a "pedo" in a tweet.[424] In September, Unsworth filed a defamation suit seeking $190 million in damages.[425][426] Musk apologized and a jury later found Musk was not liable for the claim.[427][428]

In September 2018, Musk was sued by the U.S. Securities and Exchange Commission (SEC) for a tweet stating that funding had been secured for potentially taking Tesla private.[142][h] The securities fraud lawsuit characterized the tweet as false, misleading, and damaging to investors, and sought to bar Musk from being CEO of publicly traded companies.[142][432][433] Two days later, Musk settled with the SEC, without admitting or denying the SEC's allegations. As a result, Musk and Tesla were fined $20 million each, and Musk was forced to step down for three years as Tesla chairman but was able to remain as CEO.[143] Shareholders filed a lawsuit over the tweet,[434] and in February 2023, a jury found Musk and Tesla not liable.[435] Musk has stated in interviews that he does not regret posting the tweet that triggered the SEC investigation.[436][437]

In 2019, Musk stated in a tweet that Tesla would build half a million cars that year.[438] The SEC reacted by asking a court to hold him in contempt for violating the terms of the 2018 settlement agreement. A joint agreement between Musk and the SEC eventually clarified the previous agreement details,[439] including a list of topics about which Musk needed preclearance.[440] In 2020, a judge blocked a lawsuit that claimed a tweet by Musk regarding Tesla stock price ("too high imo") violated the agreement.[441][442] Freedom of Information Act (FOIA)-released records showed that the SEC concluded Musk had subsequently violated the agreement twice by tweeting regarding "Tesla's solar roof production volumes and its stock price".[443]

In October 2023, the SEC sued Musk over his refusal to testify a third time in an investigation into whether he violated federal law by purchasing Twitter stock in 2022.[444][445][446] In February 2024, Judge Laurel Beeler ruled that Musk must testify again.[447] In January 2025, the SEC filed a lawsuit against Musk for securities violations related to his purchase of Twitter.[448]

In January 2024, Delaware judge Kathaleen McCormick ruled in a 2018 lawsuit that Musk's $55 billion pay package from Tesla be rescinded.[449] McCormick called the compensation granted by the company's board "an unfathomable sum" that was unfair to shareholders.[450] The Delaware Supreme Court overturned McCormick's decision in December 2025, restoring Musk's compensation package and awarding $1 in nominal damages.[451] Musk exercised the stock options from his 2018 pay package on June 16, 2026.[452][453]

On March 20, 2026, a jury in San Francisco, California, found Musk liable for misleading investors during his acquisition of Twitter, though they absolved him of some fraud allegations, finding he did not "scheme" to defraud investors.[454][455] Specifically, Musk was found to have misled shareholders in two tweets, but not in a podcast, as the views he expressed in that were deemed by the jury to be an opinion.[454][455] Damages have yet to be calculated, but plaintiffs' lawyers estimated them to be $2.5 billion.[456]

On May 18, 2026, a jury in Oakland, California, decided in favor of OpenAI and its executives, finding that Musk's February 2024 claims that the company violated its founding agreement by prioritizing profits over AI safety[457] were barred by the statute of limitations.[458]

Personal life

Official Royal Society portrait, 2018
Musk became a U.S. citizen in 2002.[52] From the early 2000s until late 2020, Musk resided in California, where both Tesla and SpaceX were founded.[459] He then relocated to Cameron County, Texas,[460][461] saying that California had become "complacent" about its economic success.[459][462][463]

While hosting Saturday Night Live in 2021, Musk stated that he has Asperger syndrome.[464] He was not diagnosed as a child.[465] When asked about his experience growing up with Asperger's syndrome the following year, Musk stated that "the social cues were not intuitive ... I would just tend to take things very literally ... but then that turned out to be wrong—[people were not] simply saying exactly what they mean, there's all sorts of other things that are meant, and [it] took me a while to figure that out."[466] Musk suffers from back pain and has undergone several spine-related surgeries, including a disc replacement.[467][468] In 2000, he contracted a severe case of malaria while on vacation in South Africa.[469]

Musk has stated he uses doctor-prescribed ketamine for occasional depression and that he doses "a small amount once every other week or something like that";[470] since January 2024, some media outlets have reported that he takes ketamine, marijuana, LSD, ecstasy, mushrooms, cocaine and other drugs. Musk at first refused to comment on his alleged drug use, before responding that he had not tested positive for drugs, and that if drugs somehow improved his productivity, "I would definitely take them!"[471] The New York Times' investigations revealed Musk's overuse of ketamine and numerous other drugs, as well as strained family relationships and concerns from close associates who have become troubled by his public behavior as he became more involved in political activities and government work.[472] According to The Washington Post, President Trump described Musk as "a big-time drug addict".[473]

Through his own label Emo G Records, Musk released a rap track, "RIP Harambe", on SoundCloud in March 2019.[474][475][476] The following year, he released an EDM track, "Don't Doubt Ur Vibe", featuring his own lyrics and vocals.[477]

Musk plays video games, which he stated has a "'restoring effect' that helps his 'mental calibration'".[478] Some games he plays include Quake, Diablo IV, Elden Ring, and Polytopia.[479][480] Musk once claimed to be one of the world's top video game players but has since admitted to "account boosting", or cheating by hiring outside services to achieve top player rankings.[481][482][483] Musk has justified the boosting by claiming that all top accounts do it so he has to as well to remain competitive.[484][483][485] In 2024 and 2025, Musk criticized the video game Assassin's Creed Shadows and its creator Ubisoft for "woke" content.[486] Musk posted to X that "DEI kills art" and specified the inclusion of the historical figure Yasuke in the Assassin's Creed game as offensive; he also called the game "terrible". Ubisoft responded by saying that Musk's comments were "just feeding hatred" and that they were focused on producing a game not pushing politics.[487][488]

Marriages, relationships and children
Further information: Musk family

Musk and his son X Æ A-Xii with President Donald Trump in the Oval Office in February 2025
Musk has fathered at least 14 children, one of whom died as an infant.[489] The Wall Street Journal reported in 2025 that sources close to Musk suggest that the "true number of Musk's children is much higher than publicly known".[2] He had six children with his first wife, Canadian author Justine Wilson, whom he met while attending Queen's University in Ontario, Canada; they married in 2000.[490] In 2002, their first child, Nevada Alexander Musk, died of sudden infant death syndrome at the age of 10 weeks.[491][492] After his death, the couple used in vitro fertilization (IVF) to continue their family;[492] they had twins in 2004, followed by triplets in 2006.[492] The couple divorced in 2008 and have shared custody of their children.[493][494] The elder twin he had with Wilson came out as a trans woman and, in 2022, officially changed her name to Vivian Jenna Wilson,[495] adopting her mother's surname because she no longer wished to be associated with Musk.[495]

Musk began dating English actress Talulah Riley in 2008.[496] They married two years later at Dornoch Cathedral in Scotland.[497][498] In 2012, the couple divorced, then remarried the following year.[499] After briefly filing for divorce in 2014,[499] Musk finalized a second divorce from Riley in 2016.[500] Musk then dated the American actress Amber Heard for several months in 2017;[501] he had reportedly been "pursuing" her since 2012.[502]

In 2018, Musk and Canadian musician Grimes confirmed they were dating.[503] Grimes and Musk have three children, born in 2020, 2021, and 2022.[i] Musk and Grimes originally gave their eldest child the name "X Æ A-12", which would have violated California regulations as it contained characters that are not in the modern English alphabet;[508][509] the names registered on the birth certificate are "X" as a first name, "Æ A-Xii" as a middle name, and "Musk" as a last name.[510][511] They received criticism for choosing a name perceived to be impractical and difficult to pronounce;[512] Musk has said the intended pronunciation is "X Ash A Twelve".[511] Their second child was born via surrogacy.[513] Despite the pregnancy, Musk confirmed reports that the couple were "semi-separated" in September 2021; in an interview with Time in December 2021, he said he was single.[514][515] In October 2023, Grimes sued Musk over parental rights and custody of X Æ A-Xii.[516][517][518] Musk has taken X Æ A-Xii to multiple official events in Washington, D.C. during Trump's second term in office.[519]

Also in July 2022, The Wall Street Journal reported that Musk allegedly had an affair with Nicole Shanahan, the wife of Google co-founder Sergey Brin, in 2021, leading to their divorce the following year.[520] Musk denied the report.[521] Musk also had a relationship with Australian actress Natasha Bassett, who has been described as "an occasional girlfriend".[522] In October 2024, The New York Times reported Musk bought a Texas compound for his children and their mothers,[523] though Musk denied having done so.[524]

Musk also has four children with Shivon Zilis, director of operations and special projects at Neuralink: twins born via IVF in 2021, a child born in 2024 via surrogacy and a child born in 2025.[j] In July 2026 Musk described Zilis as his partner.[531]

On February 14, 2025, Ashley St. Clair, an influencer and author, posted on X claiming to have given birth to Musk's son Romulus five months earlier, which media outlets reckoned as Musk's thirteenth child.[k] On February 22, 2025, it was reported that St Clair had filed for sole custody of her five-month-old son and for Musk to be recognised as the child's father.[535][536] On March 31, 2025, Musk wrote that, while he was unsure if he was the father of St. Clair's child, he had paid St. Clair $2.5 million and would continue paying her $500,000 per year.[l] Later reporting from the Wall Street Journal indicated that $1 million of these payments to St. Clair were structured as a loan.[2]
    """
    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them."""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOpenAI(temperature=0, model="gpt-5")
    # llm = ChatOllama(temperature = 0, model="gemma3:270m")

    chain = (
        summary_prompt_template | llm
    )  # LangChain expression language (LCEL), creates a chain by composing two components
    # Connects the output of the left component to the input of the right component
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()