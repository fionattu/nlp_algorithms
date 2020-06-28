def retrieve_entity(tag_ids):
    result, entity = [], []
    for i in range(len(tag_ids)):
        tag = tag_ids[i]  # like 'B_ns'
        if tag[0] == 'B':
            entity = [tag[2:], i, i]  # like ('ns', 2, 2)
        elif tag[0] == 'M' and len(entity) > 0 and entity[0] == tag[2:]:
            entity[2] = i
        elif tag[0] == 'E' and len(entity) > 0 and entity[0] == tag[2:]:
            entity[2] = i
            result.append(entity)
            entity = []
        else:
            if len(entity) > 0:
                result.append(entity)
                entity = []
    return result


def get_metrics(pred_tags, true_tags):
    pred_tags = retrieve_entity(pred_tags)
    true_tags = retrieve_entity(true_tags)
    union = [i for i in pred_tags if i in true_tags]  # TP
    precision = 1.0 * len(union) / len(pred_tags) if len(pred_tags) != 0 else 0.0
    recall = 1.0 * len(union) / len(true_tags) if len(true_tags) != 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if len(union) != 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1_score}

