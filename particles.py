import logging
import operator

from deap import creator
from numpy import ceil, floor, maximum, minimum
from numpy.random import uniform


def createUniformSpace(particles, designSpace):
    pointPerDimensions = 5
    valueGrid = mgrid[designSpace[0]['min']:designSpace[0]['max']:
                      complex(0, pointPerDimensions),
                      designSpace[1]['min']:designSpace[1]['max']:
                      complex(0, pointPerDimensions)]

    for i in [0, 1]:
        for j, part in enumerate(particles):
            part[i] = valueGrid[i].reshape(1, -1)[0][j]


def filterParticles(particles, designSpace):
    for particle in particles:
        filterParticle(particle, designSpace)


def filterParticle(p, designSpace):
    p.pmin = [dimSetting['min'] for dimSetting in designSpace]
    p.pmax = [dimSetting['max'] for dimSetting in designSpace]

    for i, val in enumerate(p):
        if designSpace[i]['type'] == 'discrete':
            if uniform(0.0, 1.0) < (p[i] - floor(p[i])):
                p[i] = ceil(p[i])  # + designSpace[i]['step']
            else:
                p[i] = floor(p[i])

        p[i] = minimum(p.pmax[i], p[i])
        p[i] = maximum(p.pmin[i], p[i])


def generate(designSpace):
    particle = [uniform(dimSetting['min'], dimSetting['max'])
                for dimSetting
                in designSpace]
    particle = creator.Particle(particle)
    return particle


def updateParticle(part, generation, trial, conf, designSpace):
    if conf.admode == 'fitness':
        fraction = trial.fitness_counter / conf.max_fitness
    elif conf.admode == 'iter':
        fraction = generation / conf.max_iter
    else:
        raise('[updateParticle]: adjustment mode unknown.. ')

    random1 = uniform(0, conf.phi1)
    random2 = uniform(0, conf.phi2)
    u1 = [uniform(0, conf.phi1)] * len(part)
    u2 = [uniform(0, conf.phi2)] * len(part)
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, trial.best, part))

    weight = 1.0
    if conf.weight_mode == 'linear':
        weight = conf.max_weight - (conf.max_weight -
                                    conf.min_weight) * fraction
    elif conf.weight_mode == 'norm':
        weight = conf.weight
    else:
        raise('[updateParticle]: weight mode unknown.. ')
    weightVector = [weight] * len(part.speed)
    part.speed = map(operator.add,
                     map(operator.mul, part.speed, weightVector),
                     map(operator.add, v_u1, v_u2))

    if conf.applyK is True:
        phi = array(u1) + array(u1)

        XVector = (2.0 * conf.KK) / abs(2.0 - phi -
                                        sqrt(pow(phi, 2.0) - 4.0 * phi))
        part.speed = map(operator.mul, part.speed, XVector)

    if conf.mode == 'vp':
        for i, speed in enumerate(part.speed):
            speedCoeff = (conf.K - pow(fraction, conf.p)) * part.smax[i]
            if speed < -speedCoeff:
                part.speed[i] = -speedCoeff
            elif speed > speedCoeff:
                part.speed[i] = speedCoeff
            else:
                part.speed[i] = speed
    elif conf.mode == 'norm':
        for i, speed in enumerate(part.speed):
            if speed < part.smin[i]:
                part.speed[i] = part.smin[i]
            elif speed > part.smax[i]:
                part.speed[i] = part.smax[i]
    elif conf.mode == 'exp':
        for i, speed in enumerate(part.speed):
            maxVel = (1 - pow(fraction, conf.exp)) * part.smax[i]
            if speed < -maxVel:
                part.speed[i] = -maxVel
            elif speed > maxVel:
                part.speed[i] = maxVel
    elif conf.mode == 'no':
        pass
    else:
        raise('[updateParticle]: mode unknown.. ')
    part[:] = map(operator.add, part, part.speed)
