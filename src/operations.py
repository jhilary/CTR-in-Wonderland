from datetime import datetime
from ciw.storage import Metric, Prediction
from ciw.utils import get_memory_usage
from ciw.metrics import ll
from ciw.feature_types import PlainFeature


def learn(records_generator, storage, model, storage_metrics_dumping_depth, storage_model_dumping_depth):
    start_time = datetime.now()
    for record in records_generator():
        model.learn(record)
        if model.iterations % storage_metrics_dumping_depth == 0:
            metric = Metric(records_generator.counter, model.iterations, (datetime.now() - start_time).seconds, get_memory_usage(), model.progressive_validation_logloss, model.clicks, model.not_clicks) # total_seconds?
            storage.save_metrics([metric])
            storage.dump_weights_storage(model.weights_storage)
        if model.iterations % storage_model_dumping_depth == 0:
            storage.save(model)
    #save metrics at the end of all iterations
    if model.iterations % storage_metrics_dumping_depth != 0:
        metric = Metric(records_generator.counter, model.iterations, (datetime.now() - start_time).seconds, get_memory_usage(), model.progressive_validation_logloss, model.clicks, model.not_clicks) # total_seconds?
        storage.save_metrics([metric])
        storage.dump_weights_storage(model.weights_storage)
        storage.save(model)


def validate(records_generator, model, storage_predictions_dumping_depth, storage, identifier):
    start_time = datetime.now()
    result_ll = 0
    clicks = 0
    not_clicks = 0
    for record in records_generator():
        if record.label.value == 0:
            not_clicks += 1
        else:
            clicks += 1
        record.factors["BIAS"] = PlainFeature(1)
        result_ll += ll([record.label.value], [model.predict_proba(record.factors)])
        if records_generator.counter_filtered % storage_predictions_dumping_depth == 0:
            metric = Metric(records_generator.counter, records_generator.counter_filtered, (datetime.now() - start_time).seconds, get_memory_usage(), result_ll/records_generator.counter_filtered, clicks, not_clicks) # total_seconds?
            storage.save_metrics([metric], identifier)
    #save metrics at the end of all iterations
    if model.iterations % storage_predictions_dumping_depth != 0:
        metric = Metric(records_generator.counter, records_generator.counter_filtered, (datetime.now() - start_time).seconds, get_memory_usage(), result_ll/records_generator.counter_filtered, clicks, not_clicks) # total_seconds?
        storage.save_metrics([metric], identifier)


def predict(records_generator, model, storage, identifier):
    for record in records_generator():
        record.factors["BIAS"] = PlainFeature(1)
        storage.save_predictions([Prediction(record.record_id.value, model.predict_proba(record.factors))], identifier)
