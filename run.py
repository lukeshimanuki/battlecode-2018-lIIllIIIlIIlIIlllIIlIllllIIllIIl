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
gc.queue_research(bc.UnitType.Healer)
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

	if len(criteria) == 0:
		return random.choice(directions)

	if not criteria[0]:
		return pickMoveDirection(directions, criteria[1:])

	scores = [criteria[0](direction) for direction in directions]
	best_score = max(scores)
	best_directions = [
		direction
		for direction, score in zip(directions, scores)
		if score == best_score
	]

	return pickMoveDirection(best_directions, criteria[1:])

Target = collections.namedtuple('Target', 'location value units time valid')
#TargetTypes = enum.IntEnum('TargetTypes', 'AS')
targets = []

estructures = dict()

if gc.planet() == bc.Planet.Earth:
	estructures.update({
		(unit.location.map_location().x, unit.location.map_location().y): unit
		for unit in gc.starting_map(bc.Planet.Earth).initial_units
		if unit.location.is_on_map()
	})

runtimes = []

while True:
	start = time.time()

	round_num = gc.round()

	def rprint(string):
		print("{}: {}".format(round_num, string))

	try:
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
			return location[unit.id]. \
				map_location(). \
				is_within_range(unit.attack_range(), ml)

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

		def marginal_danger(unit, ml, f):
			return sum(
				marginal_danger_from(unit, ml, enemy)
				for enemy in eattackers
				if f(enemy)
			)

		def update_sunk_danger(unit, ml):
			for enemy in eattackers:
				sunk_danger[enemy.id] += marginal_danger_from(unit, ml, enemy)

		def nearby(units, ml, r, f):
			return [
				unit
				for unit in units
				if f(unit)
				and location[unit.id].is_on_map()
				and location[unit.id].map_location().is_within_range(r, ml)
			]

		def dist_to_nearest(units, ml, f):
			valid_units = [
				unit
				for unit in units
				if unit.location.is_on_map()
				and f(unit)
			]
			if len(valid_units) == 0:
				return 0
			return min(
				unit.location.map_location().distance_squared_to(ml)
				for unit in valid_units
			)

		dunits[ateam][f] = sorted(
			dunits[ateam][f],
			key=lambda u: dist_to_nearest(
				eunits,
				u.location.map_location(),
				lambda x: True
			)
		)

		for utt in [f, r, m, k, h, w, t]:
			for unit in dunits[ateam][utt]:
				if not location[unit.id].is_on_map():
					continue

				ut = unit.unit_type

				if ut == f:
					garrison_ids = unit.structure_garrison()
					for i in range(len(garrison_ids)):
						d = random.choice(directions)
						if gc.can_unload(unit.id, d):
							gc.unload(unit.id, d)
					for gunit_id in garrison_ids:
						location[gunit_id] = gc.unit(gunit_id).location

					if gc.can_produce_robot(unit.id, buildQueue[0]):
						gc.produce_robot(unit.id, buildQueue[0])
						rprint("produced a {}".format(buildQueue[0]))
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

				if ut in [k,r] and gc.is_attack_ready(unit.id):
					enemies_in_range = [
						e
						for e in eunits
						if location[unit.id].is_within_range(
							unit.attack_range(),
							location[e.id]
						)
						and health[e.id] > 0
						and (
							ut != r
							or not location[unit.id].is_within_range(
								unit.ranger_cannot_attack_range(),
								location[e.id]
							)
						)
					]
					if len(enemies_in_range) > 0:
						enemy = min(
							enemies_in_range,
							key=value
						)
						gc.attack(unit.id, enemy.id)
						health[enemy.id] = gc.unit(enemy.id).health \
							if health[enemy.id] > unit.damage() else 0

				for d in directions:
					if gc.can_blueprint(unit.id, bc.UnitType.Factory, d):
						gc.blueprint(unit.id, bc.UnitType.Factory, d)
						rprint('built a factory')
						break

				# try to replicate
				for d in directions:
					if len(dunits[ateam][w]) < min_num_workers \
						or len(dunits[ateam][w]) < \
							len(aunits) * min_worker_ratio \
					:
						if ut == w and gc.can_replicate(unit.id, d):
							gc.replicate(unit.id, d)
							new_worker = gc.sense_unit_at_location(add(unit, d))
							units.append(new_worker)
							aunits.append(new_worker)
							dunits[ateam][w].append(new_worker)
							location[new_worker.id] = new_worker.location
							health[new_worker.id] = new_worker.health
							rprint('replicated')

				if location[unit.id].is_on_map():
					# adjacent blueprints to work on
					if ut == w:
						adjacent_bp = [
							bp
							for bp in aunits
							if bp.unit_type in [f,t]
							and not bp.structure_is_built()
							and location[unit.id].
								is_adjacent_to(location[bp.id])
						]
						if len(adjacent_bp) > 0:
							bp = max(
								adjacent_bp,
								key=value
							)
							if gc.can_build(unit.id, bp.id):
								gc.build(unit.id, bp.id)
								continue

				# try to harvest
				kdirection = max(
					list(bc.Direction),
					key=lambda d: gc.karbonite_at(add(unit, d))
						if gc.can_harvest(unit.id, d)
						else -1
				)
				if gc.can_harvest(unit.id, d):
					gc.harvest(unit.id, d)

				if ut == h and gc.is_heal_ready(unit.id):
					allies_in_range = [
						a
						for a in aunits
						if location[unit.id].is_within_range(
							unit.attack_range(),
							location[a.id]
						)
						and health[a.id] > 0
						and a.unit_type in [w,h,k,r,m]
					]
					if len(allies_in_range) > 0:
						ally = max(
							allies_in_range,
							key=value
						)
						if gc.is_heal_ready(unit.id):
							gc.heal(unit.id, ally.id)
							health[ally.id] = gc.unit(ally.id).health

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
				if location[unit.id].is_on_map():
					if gc.is_move_ready(unit.id):
						direction = pickMoveDirection(directions, [
							# validity
							lambda d: gc.can_move(unit.id, d),

							# micro
							None if not (ut in [k,m,r] and aggressive) else
								lambda d: min(1, len(
									nearby(
										eunits,
										add(unit, d),
										unit.attack_range(),
										lambda e: True
									)
								)),
							lambda d: -marginal_danger(
								unit,
								add(unit, d),
								# ignore enemies with the same range
								# unless low on health
								#lambda e: not aggressive \
								#	or ut not in [r,m]
								#	or e.attack_range() != unit.attack_range()
								lambda e: True
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
								[
									a
									for a in aunits
									if a.unit_type in [w,h,k,r,m]
								],
								add(unit, d),
								lambda u: health[u.id] < u.max_health
							),
							None if ut != w else lambda d: -dist_to_nearest(
								dunits[ateam][f] + dunits[ateam][t],
								add(unit, d),
								lambda u: health[u.id] < u.max_health
							),
							None if ut not in [k,r,m] else
								lambda d: -dist_to_nearest(
									estructures.values(),
									add(unit, d),
									lambda x: True
								),
							None if ut not in [k,r,m] else
								lambda d: min(-dist_to_nearest(
									eunits,
									add(unit, d),
									lambda e: True
								), -unit.attack_range()),

							# spread
							lambda d: -len(nearby(
								aunits,
								add(unit, d),
								1,
								lambda u: u.id != unit.id)
							),

							# exploration
							lambda d: rd.dx() * d.dx() + rd.dy() * d.dy(),
						])

						if direction and gc.can_move(unit.id, direction):
							gc.move_robot(unit.id, direction)
							location[unit.id] = gc.unit(unit.id).location

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
	if len(runtimes) % 10 == 0:
		rprint("runtime: {0:.3f}".format(max(runtimes)))
		runtimes = []

	gc.next_turn()

	sys.stdout.flush()
	sys.stderr.flush()

