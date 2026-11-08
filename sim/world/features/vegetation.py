"""Vegetation: tree stands and ground cover (geometry in world/flora.py)."""
from __future__ import annotations

from .. import flora
from ..sampling import poisson_disk
from ..world import Feature

# species: (geometry function, size parameters drawn uniformly, in this order)
SPECIES = {
    "conifer":   (flora.conifer, {"height": (10, 21)}),
    "broadleaf": (flora.broadleaf_tree, {"height": (7, 13), "crown": (2.0, 3.4)}),
    "birch":     (flora.birch, {"height": (8, 14)}),
}
NARROWER = {"conifer": "broadleaf", "broadleaf": "birch"}     # crown width order, widest first


class PlantColours(Feature):
    """Seeded colour variation of bark, leaves and grass for this scene (cheap domain
    randomisation). Add it first: it registers the vegetation materials."""

    def apply(self, w) -> None:
        flora.plant_materials(w.builder, w.rng)


class Forest(Feature):
    """A mixed stand: trees at least `spacing` apart (Poisson-disk), species drawn from `mix`
    (fractions), each with its own random size. Trees keep clear of regions (`clear`), and
    near `narrow_near = (region, {species: distance})` wide crowns are swapped for narrower
    species (conifer -> broadleaf -> birch), e.g. to keep a flight corridor open."""

    def __init__(self, spacing: float = 3.3, margin: float = 3.0, tries: int = 60000, clear=(),
                 mix=(("conifer", 0.55), ("broadleaf", 0.30), ("birch", 0.15)), narrow_near=None) -> None:
        self.spacing, self.margin, self.tries, self.clear = spacing, margin, tries, list(clear)
        self.mix, self.narrow_near = list(mix), narrow_near
        self.trees: list[tuple[str, float, float]] = []

    def _pick(self, u: float) -> str:
        cum = 0.0
        for name, frac in self.mix:
            cum = round(cum + frac, 12)
            if u < cum:
                return name
        return self.mix[-1][0]

    def apply(self, w) -> None:
        flora.plant_materials(w.builder, names=flora.TREE_MATERIALS)   # unless PlantColours set them
        rng, h, mg = w.rng, w.half, self.margin

        def accept(x, y):
            return all(c.region.distance(x, y) > c.distance for c in self.clear)
        pts = poisson_disk(rng, (-h + mg, h - mg), (-h + mg, h - mg), self.spacing, tries=self.tries, accept=accept)
        for x, y in pts:
            zz = w.height(x, y)
            species = self._pick(rng.random())
            if self.narrow_near is not None:
                region, limits = self.narrow_near
                d = region.distance(x, y)
                while species in limits and d < limits[species]:
                    species = NARROWER[species]
            fn, sizes = SPECIES[species]
            fn(w.builder, rng, x, y, zz, **{k: rng.uniform(*v) for k, v in sizes.items()})
            self.trees.append((species, float(x), float(y)))


class Grass(Feature):
    """Grass tufts scattered uniformly; `fresh` fraction green, the rest dry."""

    def __init__(self, count: int = 3500, margin: float = 2.0, fresh: float = 0.7, height=(0.2, 0.5)) -> None:
        self.count, self.margin, self.fresh, self.height = count, margin, fresh, height

    def apply(self, w) -> None:
        flora.plant_materials(w.builder, names=flora.GRASS_MATERIALS)
        rng, h, mg = w.rng, w.half, self.margin
        for x, y in rng.uniform(-h + mg, h - mg, (self.count, 2)):
            flora.grass_tuft(w.builder, rng, x, y, w.height(x, y),
                             mat="grass_tuft" if rng.random() < self.fresh else "dry_grass", h=rng.uniform(*self.height))
