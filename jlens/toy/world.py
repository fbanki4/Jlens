"""A small relational world, and a corpus generator over it.

Why bother with a synthetic world? Because the interesting J-lens claims are
about *unverbalized intermediates*: on "the color of the planet fourth from the
sun is", the lens should surface ``Mars`` even though that word appears in
neither the prompt nor the answer. To test that you need ground truth about
what the intermediate is — which real corpora do not hand you.

The world has two independent relational domains so we can ask whether
concurrently-held concepts compete:

    people  --lives_in-->  cities  --city_in-->  countries  --speaks--> languages
                                                            --region--> continents
    dishes  --made_of--->  foods   --grown_in--> climates

and a set of *composed* relations that are deterministic functions of the
atomic ones:

    hails_from  = city_in  o lives_in     (person -> country)   latent: city
    tongue      = speaks   o city_in      (city   -> language)  latent: country
    lies_within = region   o city_in      (city   -> continent) latent: country
    origin      = grown_in o made_of      (dish   -> climate)   latent: food

Atomic facts are always in the training corpus. Composed facts are split: a
train fraction is stated outright, the rest is held out entirely. A model that
answers held-out composed queries correctly *must* be composing, which means
the latent entity is being computed somewhere inside — exactly the thing the
J-lens claims to be able to read.

Everything is word-level tokenized, one token per word, so a "concept" is a
single token and the plain J-lens applies without the multi-token machinery.
:mod:`jlens.phrases` then reintroduces multi-token concepts on purpose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

PEOPLE = [
    "alice", "bob", "carol", "dave", "erin", "frank", "grace", "hank",
    "irene", "jack", "karen", "leo", "mona", "ned", "olive", "pete",
    "quinn", "rosa", "sam", "tina", "umar", "vera", "wade", "xena",
    "yuri", "zoe", "abel", "bella", "cyrus", "dana", "elias", "fiona",
    "gus", "hilda", "ivan", "jana", "kurt", "lena", "milo", "nora",
    "omar", "petra", "raj", "sonia", "tomas", "ulla", "vince", "wanda",
    "arne", "brit", "cleo", "dmitri", "esme", "felix", "greta", "hugo",
    "inga", "juno", "kira", "lars", "maja", "niko", "otto", "pia",
    "rune", "sven", "tara", "ulf", "vidal", "wren", "yara", "zane",
    "anouk", "boris", "celia", "dilan", "eira", "fabio", "gitta", "hans",
    "ilse", "josef", "klara", "linus", "marit", "nadia", "oskar", "paloma",
    "remi", "sigrid", "tobias", "ursula", "viggo", "wilma", "yusuf", "zita",
]
#: Half of these are two words long *on purpose*. The tokenizer is word-level,
#: so a two-word city is a genuinely multi-token concept with no J-lens vector
#: of its own — which is the whole difficulty the phrase experiments address.
#: Countries, languages and continents stay single-token so that the *answer*
#: is always readable in one step and accuracy stays unambiguous.
CITIES = [
    "paris", "port lyon", "tokyo", "new osaka", "cairo", "old luxor",
    "lima", "santa cusco", "oslo", "north bergen", "delhi", "new mumbai",
    "quito", "san guayaquil", "sofia", "cape varna", "madrid", "east seville",
    "berlin", "saint munich", "hanoi", "little hue", "accra", "west kumasi",
]
#: Extra words needed only to spell the two-word city names.
CITY_MODIFIERS = [
    "port", "new", "old", "santa", "north", "san", "cape", "east",
    "saint", "little", "west",
]
COUNTRIES = [
    "france", "japan", "egypt", "peru", "norway", "india",
    "ecuador", "bulgaria", "spain", "germany", "vietnam", "ghana",
]
LANGUAGES = [
    "french", "japanese", "arabic", "quechua", "norwegian", "hindi",
    "spanish", "bulgarian", "castilian", "german", "vietnamese", "twi",
]
CONTINENTS = ["europe", "asia", "africa", "america"]
COUNTRY_CONTINENT = [
    "europe", "asia", "africa", "america", "europe", "asia",
    "america", "europe", "europe", "europe", "asia", "africa",
]

DISHES = [
    "stew", "pie", "soup", "salad", "curry", "roast",
    "tart", "broth", "wrap", "hash", "gratin", "chowder",
]
FOODS = ["potato", "apple", "lentil", "kale", "mango", "beef",
         "cherry", "barley", "olive", "onion", "squash", "clam"]
CLIMATES = ["temperate", "tropical", "arid", "coastal"]
FOOD_CLIMATE = ["temperate", "temperate", "arid", "coastal", "tropical", "temperate",
                "temperate", "arid", "arid", "temperate", "tropical", "coastal"]

#: Relation name -> (surface verb phrase, domain, codomain).
ATOMIC_RELATIONS = {
    "lives_in": ("lives in", "person", "city"),
    "city_in": ("sits in", "city", "country"),
    "speaks": ("speaks", "country", "language"),
    "region": ("lies in", "country", "continent"),
    "made_of": ("made of", "dish", "food"),
    "grown_in": ("grown in", "food", "climate"),
}

#: Composed relation -> (verb phrase, outer atomic, inner atomic).
#: ``composed(x) = outer(inner(x))``; ``inner(x)`` is the latent intermediate.
COMPOSED_RELATIONS = {
    "hails_from": ("hails from", "city_in", "lives_in"),
    "tongue": ("utters", "speaks", "city_in"),
    "lies_within": ("stands within", "region", "city_in"),
    "origin": ("comes from", "grown_in", "made_of"),
}

FILLER_VERBS = ["walked", "waited", "smiled", "paused", "listened", "returned"]
FILLER_NOUNS = ["morning", "river", "market", "letter", "window", "garden"]


@dataclass
class World:
    """A concrete instantiation: entity lists plus the relation tables."""

    atomic: dict[str, dict[str, str]] = field(default_factory=dict)
    entities: dict[str, list[str]] = field(default_factory=dict)
    seed: int = 0

    def composed_map(self, name: str) -> dict[str, str]:
        """Ground-truth table for a composed relation."""
        _, outer, inner = COMPOSED_RELATIONS[name]
        return {x: self.atomic[outer][self.atomic[inner][x]] for x in self.atomic[inner]}

    def latent_map(self, name: str) -> dict[str, str]:
        """For each subject, the *unverbalized* intermediate entity."""
        _, _, inner = COMPOSED_RELATIONS[name]
        return dict(self.atomic[inner])


def build_world(seed: int = 0) -> World:
    """Wire the entity lists into relation tables (deterministic in ``seed``)."""
    rng = random.Random(seed)
    entities = {
        "person": list(PEOPLE), "city": list(CITIES), "country": list(COUNTRIES),
        "language": list(LANGUAGES), "continent": list(CONTINENTS),
        "dish": list(DISHES), "food": list(FOODS), "climate": list(CLIMATES),
    }
    # Each city belongs to the country at the same index//2; each person to a
    # city, spread evenly so every city is somebody's home.
    city_in = {c: COUNTRIES[i // 2] for i, c in enumerate(CITIES)}
    shuffled_people = list(PEOPLE)
    rng.shuffle(shuffled_people)
    lives_in = {p: CITIES[i % len(CITIES)] for i, p in enumerate(shuffled_people)}
    speaks = dict(zip(COUNTRIES, LANGUAGES, strict=True))
    region = dict(zip(COUNTRIES, COUNTRY_CONTINENT, strict=True))
    made_of = dict(zip(DISHES, FOODS, strict=True))
    grown_in = dict(zip(FOODS, FOOD_CLIMATE, strict=True))
    return World(
        atomic={
            "lives_in": lives_in, "city_in": city_in, "speaks": speaks,
            "region": region, "made_of": made_of, "grown_in": grown_in,
        },
        entities=entities,
        seed=seed,
    )


def atomic_sentence(relation: str, subject: str, obj: str) -> str:
    verb, _, _ = ATOMIC_RELATIONS[relation]
    return f"{subject} {verb} {obj} ."


def composed_sentence(relation: str, subject: str, obj: str) -> str:
    verb, _, _ = COMPOSED_RELATIONS[relation]
    return f"{subject} {verb} {obj} ."


def composed_prompt(relation: str, subject: str) -> str:
    """The query form, ending right where the answer should go."""
    verb, _, _ = COMPOSED_RELATIONS[relation]
    return f"{subject} {verb}"


@dataclass
class Split:
    """Which composed facts the model is allowed to see."""

    train: dict[str, list[str]]
    heldout: dict[str, list[str]]


def make_split(world: World, *, heldout_frac: float = 0.25, seed: int = 0) -> Split:
    """Hold out a fraction of the subjects of each composed relation."""
    rng = random.Random(seed + 991)
    train, heldout = {}, {}
    for name in COMPOSED_RELATIONS:
        subjects = sorted(world.composed_map(name))
        rng.shuffle(subjects)
        n_out = max(1, int(round(len(subjects) * heldout_frac)))
        heldout[name] = sorted(subjects[:n_out])
        train[name] = sorted(subjects[n_out:])
    return Split(train=train, heldout=heldout)


@dataclass
class Chain:
    """One in-context two-hop item.

    ``text`` states ``k`` ``person lives in city`` facts and the matching ``k``
    ``city sits in country`` facts in shuffled order, then asks
    ``<subject> hails from``. The bindings are drawn fresh every time, so no
    amount of training lets the model memorise them: answering requires
    chaining subject -> city -> country *in context*.

    ``latent`` is the city. It is the thing we want to catch the model thinking
    about: at the query position it is neither the current token nor the
    answer, and a control item can hold the prompt's token content fixed while
    changing which city is the relevant one.
    """

    text: str
    prompt: str
    subject: str
    latent: str
    answer: str
    distractor_cities: list[str]
    distractor_countries: list[str]
    #: ``person -> city`` bindings in this item, in presentation order.
    lives: list[tuple[str, str]] = field(default_factory=list)
    #: ``city -> country`` bindings in this item.
    sits: list[tuple[str, str]] = field(default_factory=list)
    #: Order the facts were presented in, as ``("lives"|"sits", index)``.
    order: list[tuple[str, int]] = field(default_factory=list)

    @property
    def latent_is_multitoken(self) -> bool:
        return " " in self.latent

    def city_country(self, city: str) -> str | None:
        for c, k in self.sits:
            if c == city:
                return k
        return None


def _render(chain_lives, chain_sits, order, subject) -> str:
    parts = []
    for kind, i in order:
        if kind == "lives":
            p, c = chain_lives[i]
            parts.append(f"{p} lives in {c} .")
        else:
            c, k = chain_sits[i]
            parts.append(f"{c} sits in {k} .")
    return " ".join(parts) + f" {subject} hails from"


def sample_chain(rng: random.Random, world: World, *, k: int = 3) -> Chain:
    """Sample a two-hop in-context item with ``k`` competing bindings."""
    people = rng.sample(world.entities["person"], k)
    cities = rng.sample(world.entities["city"], k)
    countries = rng.sample(world.entities["country"], k)
    lives = list(zip(people, cities, strict=True))
    sits = list(zip(cities, countries, strict=True))
    order = [("lives", i) for i in range(k)] + [("sits", i) for i in range(k)]
    rng.shuffle(order)
    s = rng.randrange(k)
    prompt = _render(lives, sits, order, people[s])
    return Chain(
        text=f"{prompt} {countries[s]} .",
        prompt=prompt,
        subject=people[s],
        latent=cities[s],
        answer=countries[s],
        distractor_cities=[c for i, c in enumerate(cities) if i != s],
        distractor_countries=[c for i, c in enumerate(countries) if i != s],
        lives=lives,
        sits=sits,
        order=order,
    )


def _premise_block(chain: Chain) -> str:
    return chain.prompt.rsplit(f" {chain.subject} hails from", 1)[0]


def _append_queries(head: str, queries: list[tuple[str, str]]) -> tuple[str, list[int]]:
    """Append ``(query, answer)`` pairs and return the answer token indices.

    The word-level tokenizer maps one word to one token, so a word index *is*
    a token index and the supervision mask can be built here.
    """
    words = head.split()
    supervised: list[int] = []
    for query, answer in queries:
        words.extend(query.split())
        supervised.extend(range(len(words), len(words) + len(answer.split())))
        words.extend(answer.split())
        words.append(".")
    return " ".join(words), supervised


def recall_item(
    rng: random.Random, world: World, *, k: int = 3, n_queries: int = 3
) -> tuple[str, list[int]]:
    """A one-hop in-context retrieval item with *several* queries.

    Same premise block as a chain, then repeated re-asks of facts already
    stated: ``<person> lives in`` -> their city, ``<city> sits in`` -> its
    country.

    Two design points, both learned the hard way:

    * **Two-hop composition has a prerequisite.** A model that cannot retrieve
      one binding from context cannot chain two. Left to itself, a small
      transformer sits at chance on the two-hop task indefinitely, learning
      only the *type* of the answer (a country, not the right country).
      Supervising each hop directly builds the retrieval circuit first.
    * **Supervision density decides whether the circuit forms at all.** One
      supervised answer buried in a 40-token premise block is roughly a 1-in-40
      gradient signal, and it is not enough. Asking several queries per item —
      each one requiring the same retrieval — is the difference between the
      circuit forming in a few hundred steps and never forming. (The same model
      learns textbook induction, where every position is supervised, in 200
      steps.)
    """
    chain = sample_chain(rng, world, k=k)
    pool: list[tuple[str, str]] = []
    for p, c in chain.lives:
        pool.append((f"{p} lives in", c))
    for c, country in chain.sits:
        pool.append((f"{c} sits in", country))
    rng.shuffle(pool)
    return _append_queries(_premise_block(chain), pool[:n_queries])


def chain_item(
    rng: random.Random, world: World, *, k: int = 3, n_queries: int = 2
) -> tuple[str, list[int]]:
    """A two-hop item with several queries against one premise block."""
    chain = sample_chain(rng, world, k=k)
    pool = [
        (f"{p} hails from", chain.city_country(c))
        for p, c in chain.lives
        if chain.city_country(c) is not None
    ]
    rng.shuffle(pool)
    return _append_queries(_premise_block(chain), pool[:n_queries])


def swapped_chain(rng: random.Random, chain: Chain, world: World) -> Chain | None:
    """A minimally different item: same token inventory, different correct city.

    Repoints the subject's ``lives in`` fact at one of the distractor cities.
    Every word in the prompt is unchanged — only *which* binding is the
    relevant one — so a lens signal that tracks ``latent`` across this
    manipulation cannot be explained by the bag of tokens in the context.
    """
    if not chain.distractor_cities:
        return None
    new_city = rng.choice(chain.distractor_cities)
    answer = chain.city_country(new_city)
    if answer is None:
        return None
    lives = [
        (p, new_city if p == chain.subject else c) for p, c in chain.lives
    ]
    prompt = _render(lives, chain.sits, chain.order, chain.subject)
    return Chain(
        text=f"{prompt} {answer} .",
        prompt=prompt,
        subject=chain.subject,
        latent=new_city,
        answer=answer,
        distractor_cities=[chain.latent]
        + [c for c in chain.distractor_cities if c != new_city],
        distractor_countries=[c for c in chain.distractor_countries if c != answer]
        + [chain.answer],
        lives=lives,
        sits=chain.sits,
        order=chain.order,
    )


def filler_sentence(rng: random.Random, world: World) -> str:
    """Generic text so the fitting corpus is not purely relational. The lens is
    supposed to be fitted on a broad distribution; a corpus of nothing but
    ``X lives in Y`` would bake this world's syntax into ``J_l``."""
    who = rng.choice(world.entities["person"])
    where = rng.choice(world.entities["city"] + world.entities["continent"])
    return (
        f"{who} {rng.choice(FILLER_VERBS)} near the "
        f"{rng.choice(FILLER_NOUNS)} in {where} ."
    )


def build_corpus(
    world: World,
    split: Split,
    *,
    n_sentences: int = 200_000,
    atomic_weight: float = 3.0,
    composed_weight: float = 3.0,
    # The in-context two-hop items are off by default. They are kept because
    # the machinery is useful, but a model of this size does not learn
    # in-context retrieval from them at a reasonable budget: it converges to
    # predicting the answer's *type* (a country, not the right country) and
    # sits at chance. Parametric composition, on the other hand, generalises to
    # held-out subjects at 6-8x chance, which is what the experiments need.
    chain_weight: float = 0.0,
    recall_weight: float = 0.0,
    filler_weight: float = 1.0,
    chain_ks: tuple[int, ...] = (2, 3),
    seed: int = 0,
) -> list[tuple[str, list[int] | None]]:
    """Sample a training corpus of ``(text, supervised_token_indices)`` pairs.

    ``None`` means supervise every position; a list restricts the loss to those
    token indices.

    That distinction is not a refinement, it is what makes the task learnable
    at this scale. A retrieval item's premise block states *randomly sampled*
    bindings, so predicting them is impossible by construction; supervising
    those ~30 positions swamps the few that carry retrieval signal, and the
    model settles for learning the answer's *type* (a country, not the right
    country). Masking the premises concentrates the gradient on the hops.

    Five sentence types, and the mix matters:

    * **atomic** facts, oversampled — parametric world knowledge.
    * **composed** facts for the train subjects only — the parametric two-hop
      route, which the model may or may not learn to compose (held-out
      subjects tell us which).
    * **chain** items — the in-context two-hop task with fresh random bindings.
      This is the one that reliably trains, and the one the multi-hop lens
      experiments read from.
    * **filler** — generic text, so ``J_l`` is not fitted on a corpus of pure
      relational templates.
    """
    rng = random.Random(seed + 17)
    atomic_items = [
        (name, s, o) for name, table in world.atomic.items() for s, o in table.items()
    ]
    composed_items = [
        (name, s, world.composed_map(name)[s])
        for name in COMPOSED_RELATIONS
        for s in split.train[name]
    ]
    kinds = ["atomic", "composed", "chain", "recall", "filler"]
    weights = [atomic_weight, composed_weight, chain_weight, recall_weight,
               filler_weight]
    out: list[tuple[str, list[int] | None]] = []
    for _ in range(n_sentences):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        if kind == "atomic":
            name, s, o = rng.choice(atomic_items)
            out.append((atomic_sentence(name, s, o), None))
        elif kind == "composed":
            name, s, o = rng.choice(composed_items)
            out.append((composed_sentence(name, s, o), None))
        elif kind == "chain":
            out.append(chain_item(rng, world, k=rng.choice(chain_ks)))
        elif kind == "recall":
            out.append(recall_item(rng, world, k=rng.choice(chain_ks)))
        else:
            out.append((filler_sentence(rng, world), None))
    return out


def chain_benchmark(
    world: World, *, n: int = 200, k: int = 3, seed: int = 123
) -> list[Chain]:
    """A held-out set of chain items for evaluation (disjoint RNG stream)."""
    rng = random.Random(seed)
    return [sample_chain(rng, world, k=k) for _ in range(n)]


def vocabulary(world: World) -> list[str]:
    """Every word the corpus can contain, in a stable order."""
    words = {"."}
    for group in world.entities.values():
        for name in group:
            words.update(name.split())  # two-word cities contribute both words
    for verb, _, _ in ATOMIC_RELATIONS.values():
        words.update(verb.split())
    for verb, _, _ in COMPOSED_RELATIONS.values():
        words.update(verb.split())
    words.update(FILLER_VERBS)
    words.update(FILLER_NOUNS)
    words.update(CITY_MODIFIERS)
    words.update(["near", "the", "in"])
    return ["<pad>", "<eos>"] + sorted(words)
