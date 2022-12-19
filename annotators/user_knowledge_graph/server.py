import logging
import uuid
import os
import re

import requests
from flask import Flask, jsonify, request
from deeppavlov_kg import TerminusdbKnowledgeGraph

from common.utils import get_named_persons
from common.personal_info import my_name_is_pattern

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

CUSTOM_EL_ADD = os.getenv("CUSTOM_EL_ADD")
logger.info(f"URL of custom EL: {CUSTOM_EL_ADD}")

# read all relations & properties to add them into ontology
rel_type_dict = {}
with open("rel_list.txt", "r") as fl:
    lines = fl.readlines()
    for line in lines:
        rel, rel_type = line.strip().split()
        if rel_type == "r":
            rel_type = "relation"
        else:
            rel_type = "property"
        rel_type_dict[rel.replace("_", " ")] = rel_type

DB = "test"
TEAM = "yashkens|c77b"

graph = TerminusdbKnowledgeGraph(team=TEAM, db_name=DB)

logger.info('Graph Loaded!')

# graph.drop_database()
#
# graph.ontology.create_entity_kinds(
#     entity_kinds=["Person", "User", "Habit"],
#     parents=[None, "Person", None]
# )
# for rel in rel_type_dict:
#     if rel_type_dict[rel] == 'relation':
#         continue
#         # rel_name = '_'.join(rel.split(' ')).upper()
#         # logger.info(f'adding rel: {rel_name}')
#         # graph.ontology.create_relationship_kind(rel_name, "User")
#     else:
#         if rel == "<blank>":
#             continue
#         logger.info(f'adding property: {rel}')
#         graph.ontology.create_property_kinds_of_entity_kinds(["User"], [[rel]])
# graph.ontology.create_property_kinds_of_entity_kinds(["User"], [["name"]])
#
# logger.info('Graph Populated!')


def add_name_property(graph, user_id, names):
    """Adds User Name property."""
    graph.create_or_update_property_of_entity(
        id_=user_id,
        property_kind="name",
        property_value=names[0],
    )
    logger.info(f"I already have you in the graph! Updating your property name to {names[0]}!")


def add_any_relationship(graph, entity_kind, entity_name, rel_type, user_id, entities_with_types, ex_triplets):
    """Creates an entity and a relation between it and the User from property extraction service."""
    entity_kind = entity_kind.replace('_', '').title()
    graph.ontology.create_property_kinds_of_entity_kinds([entity_kind], [["name"]])

    message = ""
    # logger.info(f"All entity Kinds: {graph.ontology.get_all_entity_kinds()}")
    if (entity_name, entity_kind) in entities_with_types:
        new_entity_id = entities_with_types[(entity_name, entity_kind)]
        message += f"entity exists: {new_entity_id} --- "
    else:
        new_entity_id = str(uuid.uuid4())
        new_entity_id = entity_kind + '/' + new_entity_id
        logger.info(f"Entity type to add: {entity_kind}")
        graph.ontology.create_entity_kinds(entity_kinds=[entity_kind], parents=[None])
        graph.create_entity(entity_kind, new_entity_id, ["name"], [entity_name])
        message += f"Added entity {entity_name} with Kind {entity_kind}! and connected it with the User {user_id}! "

    rel_name = rel_type + f"_{entity_kind.lower()}"
    if (user_id, rel_name, new_entity_id) in ex_triplets:
        message += f"triplet exists: {(rel_name, new_entity_id)}"
    else:
        graph.ontology.create_relationship_kind("User", rel_name, entity_kind)
        graph.create_relationship(user_id, rel_name, new_entity_id)
        message += f"{entity_name} is connected with User by {rel_type} relationship."
    logger.info(message)


def add_any_property(graph, user_id, property_type, property_value):
    """Adds a property from property extraction service."""
    if property_type == "<blank>":
        property_type = "other"
    property_type = '_'.join(property_type.split(' '))
    graph.ontology.create_property_kinds_of_entity_kinds(["User"], [[property_type]])
    graph.create_or_update_property_of_entity(
        entity_id=user_id,
        property_kind=property_type,
        new_property_value=property_value,
    )
    logger.info(f"I added a property {property_type} with value {property_value}!")


def get_entity_type(attributes):
    """Extracts DBPedia type from property extraction annotator."""
    entity_info = attributes.get('entity_info', [])
    if not entity_info:
        return 'Misc'
    exact_entity_info = entity_info[list(entity_info.keys())[0]]
    finegrained = exact_entity_info.get('finegrained_types', [])
    if finegrained:
        entity_type = finegrained[0].capitalize()
        logger.info(f'Fine-grained type: {entity_type}')
        return entity_type
    return 'Misc'


def add_relations_or_properties(utt, user_id, entities_with_types, ex_triplets):
    """Chooses what to add: property, relationship or nothing."""
    no_rel_message = "No relations were found!"
    attributes = utt.get("annotations", {}).get("property_extraction", {})
    logger.info(f'Attributes: {attributes}')

    if isinstance(attributes, dict):
        attributes = [attributes]
    for attribute in attributes:
        if attribute and attribute['triplet']:
            triplet = attribute['triplet']
            if triplet['subject'] != 'user':
                logger.info(no_rel_message)
                return {}
            if 'relation' in triplet:
                entity_kind = get_entity_type(attribute)
                entity_name = triplet['object']
                relation = '_'.join(triplet['relation'].split(' ')).upper()
                add_any_relationship(graph, entity_kind, entity_name, relation, user_id, entities_with_types, ex_triplets)
                return triplet
            else:
                add_any_property(graph, user_id, triplet['property'], triplet['object'])
                return triplet
    logger.info(no_rel_message)
    return {}


def name_scenario(utt, user_id):
    """Checks if there is a Name given and adds it as a property."""
    names = get_named_persons(utt)
    if not names:
        logger.info('No names were found.')
        return {}
    logger.info(f'I found a name: {names[0]}')
    existing_ids = [user[0].get("Id") for user in graph.search_for_entities("User")]
    if user_id not in existing_ids:
        # let's hope user is telling us their name if they're new here
        # actually that's an unreal situation -- delete this part
        add_name_property(graph, user_id, names)
        result = {'subject': 'user', 'property': 'name', 'object': names[0]}
    elif my_name_is_pattern.search(utt.get("text", "")):
        # if they're not new, search for pattern
        logger.info('I am in my name is patter if')
        add_name_property(graph, user_id, names)
        result = {'subject': 'user', 'property': 'name', 'object': names[0]}
    else:
        logger.info("You are telling me someone's name, but I guess it's not yours!")
        result = {}
    return result


def get_result(request):
    """Collects all relation & property information from one utterance and adds to graph."""
    uttrs = request.json.get("utterances", [])
    utt = uttrs[0]
    annotations = uttrs[0].get("annotations", {})
    custom_el_annotations = annotations.get("custom_entity_linking", [])
    entities_with_types = {}
    found_kg_ids = []
    for entity_info in custom_el_annotations:
        if entity_info.get("entity_id_tags", []):
            entities_with_types[(entity_indo["entity_substr"], entity_info["entity_id_tags"][0])] = \
                entity_info["entity_ids"][0]
            found_kg_ids.append(entity_info["entity_ids"][0])

    logger.info(f"Text: {uttrs[0]['text']}")
    logger.info(f"Property Extraction: {annotations.get('property_extraction', [])}")

    last_utt = utt["text"]
    logger.info(f"Utterance: {last_utt}")
    if not last_utt:
        return "Empty utterance"

    user_id = str(utt.get("user", {}).get("id", ""))
    user_id = "User/" + user_id
    all_entities = graph.get_all_entities()
    existing_ids = [entity["@id"] for entity in all_entities]
    logger.info(f"Existing ids: {existing_ids}")

    kg_parser_annotations = []
    ex_triplets = []
    if user_id in existing_ids:
        entity_rel_info = graph.get_relationships_of_entities([user_id])
        for rel, objects in entity_rel_info:
            for obj in objects:
                ex_triplets.append((user_id, rel, obj))
                if obj in found_kg_ids:
                    kg_parser_annotations.append([user_id, rel, obj])
        logger.info(f"User with id {user_id} already exists!")
    else:
        graph.create_entity("User", user_id, [], [])
        logger.info(f"Created User with id: {user_id}")

    entity_detection = utt.get("annotations", {}).get("entity_detection", {})
    entities = entity_detection.get("labelled_entities", [])
    entities = [entity.get("text", "no entity name") for entity in entities]
    added = []
    name_result = {}
    if entities:
        name_result = name_scenario(utt, user_id)
    property_result = add_relations_or_properties(utt, user_id, entities_with_types, ex_triplets)
    if name_result:
        added.append(name_result)
    if property_result:
        added.append(property_result)

    all_entities_new = graph.get_all_entities()
    all_entities_new = [entity for entity in all_entities_new
                        if (entity["@id"] not in existing_ids and not entity["@id"].startswith("User/"))]

    substr_list, ids_list, tags_list = [], [], []
    for entity in all_entities_new:
        if "name" in entity:
            substr_list.append(entity["name"])
            ids_list.append(entity["@id"])
            tags_list.append(entity["@type"])
    if substr_list:
        requests.post(CUSTOM_EL_ADD, json={"entity_info": {"entity_substr": substr_list,
                                                           "entity_ids": ids_list, "tags": tags_list}})
    logger.info(f"kg_parser_annotations: {kg_parser_annotations}")

    return [{'added_to_graph': added, "triplets": kg_parser_annotations}]


@app.route("/respond", methods=["POST"])
def respond():
    result = get_result(request)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8127)
