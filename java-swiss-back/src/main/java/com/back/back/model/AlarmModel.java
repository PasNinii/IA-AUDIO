package com.back.back.model;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.springframework.hateoas.RepresentationModel;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class AlarmModel extends RepresentationModel<AlarmModel> {

    private static final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private Long id;
    private String city;
    private String country;
    private boolean status;
    private double threshold;
    private LocalDateTime updatedOn;
    private LocalDateTime createdOn;
    private String path;
    private String classe;
    private String modelType;

    public boolean getStatus() {
        return this.status;
    }

    public String getUpdatedOn() {
        return this.updatedOn.format(formatter);
    }

    public void setUpdatedOn(String updatedOn) {
        this.updatedOn = LocalDateTime.parse(updatedOn, formatter);
    }

    public String getCreatedOn() {
        return this.createdOn.format(formatter);
    }

    public void setCreatedOn(String createdOn) {
        this.createdOn = LocalDateTime.parse(createdOn, formatter);
    }
}
