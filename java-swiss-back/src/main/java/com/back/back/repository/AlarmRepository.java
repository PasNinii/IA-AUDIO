package com.back.back.repository;

import com.back.back.entity.AlarmEntity;

import org.springframework.data.repository.PagingAndSortingRepository;

public interface AlarmRepository extends PagingAndSortingRepository<AlarmEntity, Long> {}
