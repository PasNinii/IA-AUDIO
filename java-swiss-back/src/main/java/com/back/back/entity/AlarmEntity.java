package com.back.back.entity;

import java.io.Serializable;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.Table;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name="alarm")
public class AlarmEntity implements Serializable {

    private static final long serialVersionUID = 1L;
    private static final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    @Id
    @GeneratedValue
    private Long id;
    private String city;
    private String country;
    private double threshold;
    private boolean status;
    @Column(name="created_on")
    private LocalDateTime createdOn;
    @Column(name="updated_on")
    private LocalDateTime updatedOn;
    @Column(name="audio_path")
    private String path;
    private String classe;
    @Column(name="model_type")
    private String modelType;

    public boolean getStatus() {
        return this.status;
    }

    public String getUpdatedOn() {
        return this.updatedOn.format(formatter);
    }

    public void setUpdatedOn(final String updatedOn) {
        this.updatedOn = LocalDateTime.parse(updatedOn, formatter);
    }

    public String getCreatedOn() {
        return this.createdOn.format(formatter);
    }

    public void setCreatedOn(final String createdOn) {
        this.createdOn = LocalDateTime.parse(createdOn, formatter);
    }

}
