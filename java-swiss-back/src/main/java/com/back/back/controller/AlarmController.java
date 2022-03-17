package com.back.back.controller;

import java.util.List;

import com.back.back.assembler.AlarmModelAssembler;
import com.back.back.entity.AlarmEntity;
import com.back.back.model.AlarmModel;
import com.back.back.repository.AlarmRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PagedResourcesAssembler;
import org.springframework.hateoas.CollectionModel;
import org.springframework.hateoas.PagedModel;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@CrossOrigin(origins="http://localhost:4200")
@RestController
public class AlarmController {
    @Autowired
    private AlarmRepository repository;

    @Autowired
    private AlarmModelAssembler assembler;

    @Autowired
    private PagedResourcesAssembler<AlarmEntity> pagedAssembler;

    @GetMapping("alarms")
    public ResponseEntity<CollectionModel<AlarmModel>> all() {

        List<AlarmEntity> entities = (List<AlarmEntity>) repository.findAll();

        return new ResponseEntity<>(assembler.toCollectionModel(entities), HttpStatus.OK);
    }

    @GetMapping("alarm-list")
    public ResponseEntity<PagedModel<AlarmModel>> all(Pageable pageable) {

        Page<AlarmEntity> entities = repository.findAll(pageable);

        PagedModel<AlarmModel> models = pagedAssembler.toModel(entities, assembler);

        return new ResponseEntity<>(models, HttpStatus.OK);
    }

    @GetMapping("alarms/{id}")
    public ResponseEntity<AlarmModel> one(@PathVariable Long id) {
        return repository.findById(id)
            .map(assembler::toModel)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }
}
