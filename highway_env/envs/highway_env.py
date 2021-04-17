            flags=[False,True,False]
            # array=[t,None,np.random.randint(23,25)]
            for i in range(others+1):
                if i==t:
                    controlled_vehicle=self.action_type.vehicle_class.create_random(
                    self.road,
                    amir=False,
                    speed=np.random.randint(23,29),

                    lane_id=1,
                    spacing=self.config["ego_spacing"])
                    self.controlled_vehicles.append(controlled_vehicle)
                    self.road.vehicles.append(controlled_vehicle)
                else:
                  back=False
                  speed=np.random.randint(23,29)
                  self.road.vehicles.append(
                      other_vehicles_type.create_random(self.road,amir=flags[i],speed=speed,lane_id=l, spacing=1 / self.config["vehicles_density"])
                  )
