from vizier import pyvizier as vz

from problems.synthetic import SyntheticTorch


def problem_statement(search_space_id, dataset_id):
    dim = 10
    lb, ub = -5, 5
    f = SyntheticTorch(
        search_space_id, 
        dataset_id, 
        dim,
        lb,
        ub,
    )
    problem = vz.ProblemStatement()
    root = problem.search_space.root
    for i in range(f.dim):
        root.add_float_param('x{}'.format(i), lb, ub)
    metric = vz.MetricInformation(
        name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE,
    )
    problem.metric_information.append(metric)
    return problem, f
