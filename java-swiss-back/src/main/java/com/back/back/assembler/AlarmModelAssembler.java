package com.back.back.assembler;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.linkTo;
import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.methodOn;

import com.back.back.controller.AlarmController;
import com.back.back.entity.AlarmEntity;
import com.back.back.model.AlarmModel;

import org.springframework.hateoas.CollectionModel;
import org.springframework.hateoas.server.mvc.RepresentationModelAssemblerSupport;
import org.springframework.stereotype.Component;

@Component
public class AlarmModelAssembler extends RepresentationModelAssemblerSupport<AlarmEntity, AlarmModel> {

    public AlarmModelAssembler() {
        super(AlarmController.class, AlarmModel.class);
    }

    @Override
    public AlarmModel toModel(AlarmEntity entity) {
        AlarmModel alarm = instantiateModel(entity);

        alarm.add(linkTo(methodOn(AlarmController.class).all()).withRel("alarms"));
        alarm.add(linkTo(methodOn(AlarmController.class).one(entity.getId())).withSelfRel());

        alarm.setId(entity.getId());
        alarm.setCity(entity.getCity());
        alarm.setCountry(entity.getCountry());
        alarm.setThreshold(entity.getThreshold());
        alarm.setStatus(entity.getStatus());
        alarm.setCreatedOn(entity.getCreatedOn());
        alarm.setUpdatedOn(entity.getUpdatedOn());
        alarm.setPath(entity.getPath());
        alarm.setClasse(entity.getClasse());
        alarm.setModelType(entity.getModelType());

        return alarm;
    }

    @Override
    public CollectionModel<AlarmModel> toCollectionModel(Iterable<? extends AlarmEntity> entities) {
        CollectionModel<AlarmModel> alarms = super.toCollectionModel(entities);

        alarms.add(linkTo(methodOn(AlarmController.class).all()).withSelfRel());

        return alarms;
    }
}
