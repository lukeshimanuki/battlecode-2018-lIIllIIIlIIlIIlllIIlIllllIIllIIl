import battlecode as bc
#import numpy as np
import random
import time
import sys
import traceback
import collections
import enum

print("pystarting")

gc = bc.GameController()
directions = list(set(bc.Direction) - {bc.Direction.Center})

print("pystarted")

random.seed(3142)

gc.reset_research()
gc.queue_research(bc.UnitType.Ranger)
gc.queue_research(bc.UnitType.Healer)
gc.queue_research(bc.UnitType.Healer)
#gc.queue_research(bc.UnitType.Healer)
#gc.queue_research(bc.UnitType.Mage)
gc.queue_research(bc.UnitType.Rocket)
gc.queue_research(bc.UnitType.Mage)
gc.queue_research(bc.UnitType.Mage)
gc.queue_research(bc.UnitType.Mage)
gc.queue_research(bc.UnitType.Ranger)
gc.queue_research(bc.UnitType.Knight)

r = bc.UnitType.Ranger
h = bc.UnitType.Healer
m = bc.UnitType.Mage
k = bc.UnitType.Knight
w = bc.UnitType.Worker
t = bc.UnitType.Rocket
f = bc.UnitType.Factory

roam_directions = dict()
roam_time = dict()

initialBuildQueue = [r]

def normalize_ratio(ratios):
	total = sum(ratios.values())
	return {
		ut: ratios[ut] / total if ut in ratios else 0
		for ut in [h,k,r,m]
	}

def desired_unit_ratio(round_num):
	if round_num < 50:
		return normalize_ratio({
			r: 1,
		})
	else:
		return normalize_ratio({
			r: 2,
			h: 1,
		})

def hml(ml):
	return (True if ml.planet == bc.Planet.Earth else False, ml.x, ml.y)

def uhml(tup):
	p, x, y = tup
	return bc.MapLocation(bc.Planet.Earth if p else bc.Planet.Mars, x, y)

def filter(data, f):
	return [d for d in data if f]

min_num_workers = 8
min_worker_ratio = .2

aggressive_attacker_count = 30
nonaggressive_threshold = 1.0
aggressive_threshold = .5

buildQueue = collections.deque(initialBuildQueue)


ateam = gc.team()
eteam = bc.Team.Blue if ateam == bc.Team.Red else bc.Team.Red

def pickMoveDirection(directions, criteria):
	if len(directions) == 0:
		return None

	if len(directions) == 1:
		return directions[0]

	if len(criteria) == 0:
		return random.choice(directions)

	if not criteria[0]:
		return pickMoveDirection(directions, criteria[1:])

	scores = [criteria[0](direction) for direction in directions]
	best_score = max(scores)
	best_directions = [
		direction
		for direction, score in zip(directions, scores)
		if score >= float(int(best_score)) - 1e-6
	]

	return pickMoveDirection(best_directions, criteria[1:])

Target = collections.namedtuple('Target', 'location value units time valid')
#TargetTypes = enum.IntEnum('TargetTypes', 'AS')
targets = []

estructures = dict()

planet = gc.planet()
pmap = gc.starting_map(planet)

if planet == bc.Planet.Earth:
	estructures.update({
		(unit.location.map_location().x, unit.location.map_location().y): unit
		for unit in pmap.initial_units
		if unit.location.is_on_map()
	})

# get mars locations
mmap = gc.starting_map(bc.Planet.Mars)
mars_locations = []
for ml in gc.all_locations_within(
	bc.MapLocation(bc.Planet.Mars, 0, 0),
	5000
):
	if mmap.is_passable_terrain_at(ml) and all(
		not ml.is_adjacent_to(ml2)
		for ml2 in mars_locations
	):
		mars_locations.append(ml)

asteroids = dict()
if planet == bc.Planet.Mars:
	ap = gc.asteroid_pattern()
	for round_num in range(1000):
		if ap.has_asteroid(round_num):
			ast = ap.asteroid(round_num)
			asteroids[round_num] = ast

kearth = set()
for ml in gc.all_locations_within(bc.MapLocation(planet, 0, 0), 5000):
	if pmap.initial_karbonite_at(ml) > 0:
		if all(uhml(kml).distance_squared_to(ml) > 25 for kml in kearth):
			kearth.add(hml(ml))

runtimes = []

ran_out_of_time = False
printed_ran_out_of_time = False

while True:
	start = time.time()

	round_num = gc.round()

	def rprint(string):
		print("{}: {}".format(round_num, string))

	try:
		karbonite = gc.karbonite()

		units = list(gc.units())
		# a is allied team, e is enemy team
		aunits = [unit for unit in units if unit.team == ateam]
		eunits = [unit for unit in units if unit.team == eteam]

		eattackers = [
			unit
			for unit in eunits
			if unit.unit_type in [k,m,r]
			and unit.location.is_on_map()
			and unit.attack_heat() < 100
		]

		health = {unit.id: unit.health for unit in units}
		location = {unit.id: unit.location for unit in units}

		dunits = {
			team: {
				unit_type: [
					unit for unit in units
					if unit.team == team
					and unit.unit_type == unit_type
				] for unit_type in bc.UnitType
			} for team in bc.Team
		}

		munits = {
			(unit.location.map_location().x, unit.location.map_location().y):
				unit
			for unit in units
			if unit.location.is_on_map()
		}

		def ulocation(unit, loc):
			id = unit.id
			if id in location:
				ploc = location[id]
				if ploc.is_on_map():
					ml = ploc.map_location()
					munits.pop((ml.x, ml.y))
			if loc.is_on_map():
				ml = loc.map_location()
				munits[(ml.x, ml.y)] = unit
			location[id] = loc

		# remove invalid targets
		#targets = [t for t in targets if all([v(t) for v in t.valid])]
		## find new targets
		## enemy buildings to attack
		#targets += [
		#	Target(e.location.map_location(), 100, {k,r,m}, round_num, [
		#		lambda t: 
		#	]])
		#	for e in dunits[eteam][f] + dunits[eteam][t]
		#	if e.location.is_on_map()
		#]

		# remove nonexisting estructures
		to_remove = []
		for es in estructures.values():
			ml = es.location.map_location()
			loc = (ml.x, ml.y)
			if gc.can_sense_location(ml):
				if gc.has_unit_at_location(ml):
					u = gc.sense_unit_at_location(ml)
					if u.team != eteam or u.unit_type not in [f,t]:
						to_remove.append(loc)
				else:
					to_remove.append(loc)
		for ml in to_remove:
			estructures.pop(ml)
		# add new estructures
		estructures.update({
			(e.location.map_location().x, e.location.map_location().y): e
			for e in dunits[eteam][f] + dunits[eteam][t]
		})
		location.update({
			unit.id: unit.location
			for unit in estructures.values()
		})
		estructuresv = estructures.values()

		if len(dunits[ateam][w]) < min_num_workers \
			and (len(buildQueue) == 0 or buildQueue[0] != w) \
			and all([u.ability_cooldown() < 20 for u in dunits[ateam][w]]) \
		:
			buildQueue.appendleft(w)

		def add(unit, direction):
			return location[unit.id].map_location().add(direction)

		def value(unit):
			# value per health point
			# positive is good
			# negative is bad (ie enemies)
			if health[unit.id] == 0:
				return 0

			if unit.team == ateam \
				and unit.unit_type in [f,t] \
				and not unit.structure_is_built \
			:
				return health[unit.id] / unit.max_health + 1

			return unit.max_health / max(
				health[unit.id],
				unit.max_health / 2
			) * (1 if unit.team == ateam else -1)

		def can_attack(unit, ml):
			if unit.unit_type not in [k,m,r]:
				return False
			if not location[unit.id].is_on_map():
				return False
			uml = location[unit.id].map_location()
			return uml.is_within_range(unit.attack_range(), ml) and (
				unit.unit_type != r or
				not uml.is_within_range(unit.ranger_cannot_attack_range(), ml)
			)

		sunk_danger = {unit.id: 0 for unit in eattackers}

		def marginal_danger_from(unit, ml, enemy):
			#return max(0,
			#	value(unit) *
			#	max(unit.health, enemy.damage()) * (
			#		1 if can_attack(enemy, ml) else
			#		.5 if any(
			#			can_attack(enemy, ml.add(d))
			#			for d in bc.Direction
			#			) else 0) -
			#		sunk_danger[enemy.id]
			return max(0,
				value(unit) *
				max(unit.health, enemy.damage()) *
				(1 if can_attack(enemy, ml) else 0) -
				sunk_danger[enemy.id]
			)

		def marginal_danger(unit, ml, f, enemies=eattackers):
			return sum(
				marginal_danger_from(unit, ml, enemy)
				for enemy in enemies
				#if f(enemy) and can_attack(enemy, ml)
			)

		def update_sunk_danger(unit, ml):
			for enemy in eattackers:
				sunk_danger[enemy.id] += marginal_danger_from(unit, ml, enemy)

		def exists_nearby(units, ml, r, f):
			for unit in units:
				if f(unit) and \
					location[unit.id].is_on_map() and \
					location[unit.id].map_location().is_within_range(r, ml) \
				:
					return True
			return False

		def nearby(units, ml, r, f):
			return [
				unit
				for unit in units
				if f(unit)
				and location[unit.id].is_on_map()
				and location[unit.id].map_location().is_within_range(r, ml)
			]

		def dist_to_nearest(units, ml, f, stop_min_dist=0):
			min_dist = None
			for unit in units:
				if location[unit.id].is_on_map() and f(unit):
					dist = location[unit.id]. \
						map_location(). \
						distance_squared_to(ml)
					if not min_dist or dist < min_dist:
						min_dist = dist
						if dist < stop_min_dist:
							return dist

			if not min_dist:
				return 0

			return min_dist

		def adjacent(ml, r, f):
			x, y = ml.x, ml.y
			loc = [
				(x + 1, y),
				(x - 1, y),
				(x, y + 1),
				(x, y - 1),
			] + ([
				(x + 1, y + 1),
				(x + 1, y - 1),
				(x - 1, y + 1),
				(x - 1, y - 1),
			] if r == 2 else [])

			return [
				munits[um]
				for um in loc
				if um in munits
				and f(munits[um])
			]

		def dot(d1, d2):
			return d1.dx() * d2.dx() + d1.dy() * d2.dy()

		def centroid(units):
			tx = 0
			ty = 0
			num = 0
			for unit in units:
				if unit.location.is_on_map():
					num += 1
					ml = unit.location.map_location()
					tx += ml.x
					ty += ml.y
			if num == 0:
				return bc.MapLocation(planet, 0, 0)

			return bc.MapLocation(planet, int(tx / num), int(ty / num))

		acentroid = centroid(aunits)
		ecentroid = centroid(eunits + list(estructures.values()))

		dunits[ateam][w] = sorted(sorted(dunits[ateam][w],
			# closer to enemies
			key=lambda a: a.location.
				map_location().
				distance_squared_to(ecentroid)
				if a.location.is_on_map() else float('inf')
			),
			# next tu structures
			key=lambda a: -len(adjacent(a.location.map_location(), 2,
				lambda u: u.unit_type in [f,t] and u.health < u.max_health
			)) if a.location.is_on_map() else float('-inf')
		)

		dunits[ateam][f] = sorted(dunits[ateam][f], key=lambda a:
			a.location.map_location().distance_squared_to(ecentroid)
			if a.location.is_on_map() else float('inf')
		)

		gtm = gc.research_info().get_level(t) >= 1 and (
			#gc.winning_team() == ateam or
			(
				len(dunits[ateam][f]) >= 2 and
				len(dunits[ateam][w]) >= min_num_workers and
				sum(len(dunits[ateam][ut]) for ut in [k,m,r]) >
					sum(len(dunits[eteam][ut]) for ut in [k,m,r])
			) or (
				sum(len(dunits[ateam][ut]) for ut in [k,m,r]) >
					aggressive_attacker_count
			)
		)

		# sort enemies in order of who we want to attack
		veunits = sorted(
			[e for e in eunits if location[e.id].is_on_map()],
			key=value
		)

		# sort allies in order of who we want to heal
		hunits = sorted(
			[
				a for a in aunits
				if a.unit_type in [w,h,k,r,m]
				and a.location.is_on_map()
				and health[a.id] < a.max_health
			],
			key=lambda a: -value(a)
		)

		bunits = [
			u
			for u in dunits[ateam][f] + dunits[ateam][t]
			if u.health < u.max_health
		]

		deposits = [
			ast.location
			for rn, ast in asteroids.items()
			if rn <= round_num
			and gc.can_sense_location(ast.location)
			and gc.karbonite_at(ast.location) > 0
		] if planet == bc.Planet.Mars else [
			uhml(ml)
			for ml in kearth
			if gc.can_sense_location(uhml(ml))
		]
		to_remove = {
			ml
			for ml in kearth
			if gc.can_sense_location(uhml(ml))
			and gc.karbonite_at(uhml(ml)) == 0
		}
		kearth -= to_remove

		for utt in [t, f, r, m, k, h, w]:
			if gc.get_time_left_ms() < 1000:
				ran_out_of_time = True
				continue

			for unit in dunits[ateam][utt]:
				if not location[unit.id].is_on_map():
					continue

				ut = unit.unit_type

				if ut == t:
					if planet == bc.Planet.Earth:
						for a in adjacent(
							location[unit.id].map_location(),
							2,
							lambda a: a.id != unit.id and a.team == ateam
						):
							if gc.can_load(unit.id, a.id):
								gc.load(unit.id, a.id)
								ulocation(a, gc.unit(a.id).location)

						if len(unit.structure_garrison()) > 0 \
							and len(mars_locations) > 0 \
						:
							ml = mars_locations[-1]
							if gc.can_launch_rocket(unit.id, ml):
								gc.launch_rocket(unit.id, ml)
								mars_locations.pop()
					else:
						udirections = sorted(directions, key=lambda d:
							add(unit, d).distance_squared_to(ecentroid)
						)
						for d in udirections:
							if gc.can_unload(unit.id, d):
								gc.unload(unit.id, d)
								uu = gc.sense_unit_at_location(add(unit, d))
								ulocation(uu, uu.location)


				if ut == f:
					udirections = sorted(directions, key=lambda d:
						add(unit, d).distance_squared_to(ecentroid)
					)
					for d in udirections:
						if gc.can_unload(unit.id, d):
							gc.unload(unit.id, d)
							uu = gc.sense_unit_at_location(add(unit, d))
							ulocation(uu, uu.location)

					if gc.can_produce_robot(unit.id, buildQueue[0]) \
						and not gtm \
						and sum(len(dunits[ateam][ut]) for ut in [k,m,r]) < \
							aggressive_attacker_count \
					:
						gc.produce_robot(unit.id, buildQueue[0])
						karbonite -= buildQueue[0].factory_cost()
						#rprint("produced a {}".format(buildQueue[0]))
						buildQueue.popleft()
						if len(buildQueue) == 0:
							ratio = {
								ut: len(dunits[ateam][ut])
								for ut in [h,r,m,k]
							}
							dratio = desired_unit_ratio(round_num)
							frac = {
								ut: ratio[ut] / dratio[ut]
								if dratio[ut] > 0
								else float('inf')
								for ut in [h,r,m,k]
							}
							next_unit = min([r,h,m,k], key=lambda ut: frac[ut])
							buildQueue.append(next_unit)

				if ut in [k,r,m] and gc.is_attack_ready(unit.id):
					for enemy in veunits:
						if can_attack(unit, location[enemy.id].map_location()) \
							and health[enemy.id] > 0 \
						:
							gc.attack(unit.id, enemy.id)
							health[enemy.id] = gc.unit(enemy.id).health \
								if health[enemy.id] > unit.damage() else 0
							break

				# try to replicate
				nearby_bps = []
				if ut == w and ( \
					len(dunits[ateam][f]) > 0 \
					or planet == bc.Planet.Mars \
				) and ( \
					len(dunits[ateam][w]) < min_num_workers \
					or len(dunits[ateam][w]) < \
						len(aunits) * min_worker_ratio \
				):
					nearby_bps = adjacent(
						location[unit.id].map_location(),
						2,
						lambda u: u.unit_type in [f,t] and
							not u.structure_is_built() and
							u.team == ateam
					)
					rdirections = directions
					if len(nearby_bps) > 0:
						bp = nearby_bps[0]
						ml = location[bp.id].map_location()
						bpdir = location[unit.id]. \
							map_location(). \
							direction_to(ml)
						odir = bpdir.opposite()
						rdirections = [
							bpdir.rotate_right().rotate_right(),
							bpdir.rotate_right(),
							bpdir.rotate_left().rotate_left(),
							bpdir.rotate_left(),
							odir.rotate_right(),
							odir.rotate_left(),
							odir,
						] if not bpdir.is_diagonal() else [
							bpdir.rotate_right(),
							bpdir.rotate_left(),
							bpdir.rotate_right().rotate_right(),
							bpdir.rotate_left().rotate_left(),
							odir.rotate_right(),
							odir.rotate_left(),
							odir,
						]
					for d in rdirections:
						if ut == w and gc.can_replicate(unit.id, d):
							gc.replicate(unit.id, d)
							new_worker = gc.sense_unit_at_location(
								add(unit, d)
							)
							units.append(new_worker)
							aunits.append(new_worker)
							dunits[ateam][w].append(new_worker)
							ulocation(new_worker, new_worker.location)
							health[new_worker.id] = new_worker.health
							karbonite -= w.replicate_cost()
							#rprint('replicated')

				# adjacent blueprints to work on
				if ut == w:
					adjacent_bp = adjacent(location[unit.id].map_location(), 2,
						lambda a: a.unit_type in [f, t] and
							not a.structure_is_built()
					)
					if len(adjacent_bp) > 0:
						bp = max(
							adjacent_bp,
							key=value
						)
						if gc.can_build(unit.id, bp.id):
							gc.build(unit.id, bp.id)

				# build factory / rocket
				if ut == w and karbonite > min( \
					f.blueprint_cost(), \
					t.blueprint_cost() \
				):
					bdirections = sorted(directions, key=lambda d: -len([
						dd
						for dd in directions
						if pmap.on_map(add(unit, d).add(dd))
						and gc.can_sense_location(add(unit, d).add(dd))
						and gc.is_occupiable(add(unit, d).add(dd))
					]))
					for d in bdirections:
						if gc.can_blueprint(unit.id, t, d) and gtm:
							gc.blueprint(unit.id, t, d)
							karbonite -= t.blueprint_cost()
							#rprint('built a rocket')
							break
						elif gc.can_blueprint(unit.id, f, d):
							gc.blueprint(unit.id, f, d)
							karbonite -= f.blueprint_cost()
							#rprint('built a factory')
							break

				# try to harvest
				if ut == w:
					kdirection = max(
						list(bc.Direction),
						key=lambda d: gc.karbonite_at(add(unit, d))
							if gc.can_harvest(unit.id, d)
							else -1
					)
					if gc.can_harvest(unit.id, d):
						gc.harvest(unit.id, d)

				if ut == h and gc.is_heal_ready(unit.id):
					for ally in hunits:
						if location[unit.id].is_within_range( \
							unit.attack_range(), \
							location[ally.id] \
						) and health[ally.id] > 0:
							gc.heal(unit.id, ally.id)
							health[ally.id] = gc.unit(ally.id).health
							break

				# update roam
				if unit.id not in roam_directions or roam_time[unit.id] < 1:
					roam_directions[unit.id] = random.choice(directions)
					roam_time[unit.id] = 20
				else:
					roam_time[unit.id] -= 1
				rd = roam_directions[unit.id]

				dmg_thresh = nonaggressive_threshold if \
					len(dunits[ateam][r]) + \
					len(dunits[ateam][m]) + \
					len(dunits[ateam][k]) < aggressive_attacker_count \
					else aggressive_threshold
				aggressive = health[unit.id] >= unit.max_health * dmg_thresh

				# try to move
				# greedy minimize marginal danger
				# given movements of prior robots
				uml = location[unit.id].map_location()
				if gc.is_move_ready(unit.id) \
					and (ut != w or len(nearby_bps) == 0) \
				:
					direction = pickMoveDirection(directions, [
						# validity
						lambda d: gc.can_move(unit.id, d),

						# micro
						None if not (ut in [k,m,r] and aggressive) else
							lambda d: exists_nearby(
								eunits,
								add(unit, d),
								unit.attack_range(),
								lambda e: True
							),
						lambda d, ff=filter(
							eattackers,
							lambda e: location[e.id].is_within_range(
								(e.attack_range()**.5 + 2) ** 2,
								location[unit.id]
							) and not location[e.id].is_within_range(
								max(0, (e.attack_range()**.5 - 2)) ** 2,
								location[unit.id]
							)
						): -marginal_danger(
							unit,
							add(unit, d),
							# ignore enemies with the same range
							# unless low on health
							#lambda e: not aggressive \
							#	or ut not in [r,m]
							#	or e.attack_range() != unit.attack_range()
							lambda e: True,
							ff
						),
						None if ut in [k,m,r] and aggressive else lambda d:
							dist_to_nearest(
								eattackers,
								add(unit, d),
								lambda e:
									e.
									location.
									map_location().
									is_within_range(
										int((e.attack_range()**.5 + 3)**2),
										add(unit, d)
									)
							),
						#None if ut not in [k,r,m] else
						#	lambda d: dist_to_nearest(
						#		eattackers,
						#		add(unit, d),
						#		lambda u: can_attack(u, add(unit, d))
						#	),

						# macro
						None if ut != h else lambda d: -dist_to_nearest(
							hunits,
							add(unit, d),
							lambda u: True
						),
						None if ut != w else lambda d: -dist_to_nearest(
							bunits,
							add(unit, d),
							lambda u: True
						),
						None if ut not in [k,r,m] else
							lambda d: -dist_to_nearest(
								estructuresv,
								add(unit, d),
								lambda x: True
							),
						None if ut not in [k,r,m] else
							lambda d: min(-dist_to_nearest(
								eunits,
								add(unit, d),
								lambda e: True,
								unit.attack_range()
							), -unit.attack_range()),
						None if ut != w else
							lambda d: min([
								location[unit.id].
								map_location().
								distance_squared_to(ml)
								for ml in deposits
							] + [float('5000')]),

						# spread
						lambda d: -len(adjacent(
							add(unit, d),
							1,
							lambda u: u.id != unit.id and u.team == ateam
						)),

						# exploration
						lambda d: dot(rd, d)
					])

					if direction and gc.can_move(unit.id, direction):
						gc.move_robot(unit.id, direction)
						ulocation(unit, gc.unit(unit.id).location)

					if not direction or \
						direction.dx() * rd.dx() + \
						direction.dy() * rd.dy() < 0 \
					:
						roam_directions[unit.id] = random.choice(directions)
						roam_time[unit.id] = 20
				update_sunk_danger(unit, location[unit.id].map_location())

	except Exception as e:
		print('error:', e)
		traceback.print_exc()

	end = time.time()
	runtimes.append(end - start)
	if len(runtimes) % 100 == 0:
		rprint("runtime: {0:.3f}".format(max(runtimes)))
		rprint("time remaining: {}".format(gc.get_time_left_ms()))
		#runtimes = []

	if ran_out_of_time and not printed_ran_out_of_time:
		printed_ran_out_of_time = True
		rprint('##### RAN OUT OF TIME #####')

	gc.next_turn()

	sys.stdout.flush()
	sys.stderr.flush()

