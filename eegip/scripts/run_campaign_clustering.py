from eegip.campaign import Campaign
from collections import OrderedDict


small = False
resume = True
config = '/home/oreillyc/Mayada-Python/eegip/config_clustering_study.yaml'

job_parameters_dict = OrderedDict((
    ("preprocess", {"candidate_properties": ["dataset"],
                    "fct_str": "preprocess_dataset",
                    "import_module": "eegip.preprocessing"}),
    ("compute_sources", {"candidate_properties": ["dataset", "inv_method"],
                         "fct_str": "compute_dataset_sources",
                         "import_module": "eegip.sources"}),
    ("comp_mat_con", {"candidate_properties": ["dataset", "inv_method", "con_type", "method"],
                      "fct_str": "compute_connectivity_matrices",
                      "import_module": "eegip.connectivity"}),
    ("comp_mat_agg", {"candidate_properties": ["dataset", "inv_method", "con_type", "method"],
                      "fct_str": "compute_connectivity_aggregate",
                      "import_module": "eegip.analyses"})))

job_dependencies = {"compute_sources": ["preprocess"],
                    "comp_mat_con": ["compute_sources"],
                    "comp_mat_agg": ["comp_mat_con"]}


def main():
    campaign = Campaign(small, resume, config, name="clustering")
    campaign.run(job_parameters_dict, job_dependencies, include=["comp_mat_agg"])
    campaign.save()


if __name__ == "__main__":
    main()