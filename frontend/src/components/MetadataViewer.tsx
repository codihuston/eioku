import { useEffect, useState } from 'react';

interface Metadata {
  latitude?: number;
  longitude?: number;
  altitude?: number;
  image_size?: string;
  megapixels?: number;
  rotation?: number;
  avg_bitrate?: string;
  duration_seconds?: number;
  frame_rate?: number;
  codec?: string;
  file_size?: number;
  file_type?: string;
  mime_type?: string;
  camera_make?: string;
  camera_model?: string;
  create_date?: string;
}

interface LocationData {
  latitude: number;
  longitude: number;
  altitude?: number;
  country?: string;
  state?: string;
  city?: string;
}

interface Artifact {
  artifact_id: string;
  payload: Metadata;
}

interface Props {
  videoId: string;
  apiUrl?: string;
}

export default function MetadataViewer({ videoId, apiUrl = 'http://localhost:8080' }: Props) {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [location, setLocation] = useState<LocationData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch metadata artifacts
    fetch(`${apiUrl}/api/v1/videos/${videoId}/artifacts?type=video.metadata`)
      .then(res => res.json())
      .then((data: Artifact[]) => {
        if (data && data.length > 0) {
          setMetadata(data[0].payload);
        }
      })
      .catch(() => {
        // Metadata is optional, continue
      });

    // Fetch location data from projection
    fetch(`${apiUrl}/api/v1/videos/${videoId}/location`)
      .then(res => {
        if (res.ok) return res.json();
        return null;
      })
      .then(data => {
        if (data) {
          setLocation(data);
        }
        setLoading(false);
      })
      .catch(() => {
        // Location data is optional, don't set error
        setLoading(false);
      });
  }, [videoId, apiUrl]);

  const formatGPS = (lat: number, lon: number): string => {
    const latDir = lat >= 0 ? 'N' : 'S';
    const lonDir = lon >= 0 ? 'E' : 'W';
    return `${Math.abs(lat).toFixed(4)}°${latDir}, ${Math.abs(lon).toFixed(4)}°${lonDir}`;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    try {
      return new Date(dateString).toLocaleString(undefined, {
        dateStyle: 'short',
        timeStyle: 'short',
      });
    } catch {
      return dateString;
    }
  };

  if (loading) {
    return <div style={{ padding: '20px' }}>Loading metadata...</div>;
  }

  if (!metadata && !location) {
    return <div style={{ padding: '20px', color: '#999' }}>No metadata available</div>;
  }

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Location Section (from projection) */}
        {location && (
          <div>
            <h3 style={{ margin: '0 0 12px 0', color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              🌍 Location
            </h3>
            <div
              style={{
                padding: '12px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                border: '1px solid #444',
              }}
            >
              <div style={{ fontSize: '14px', color: '#1976d2', marginBottom: '8px', fontWeight: '500' }}>
                {formatGPS(location.latitude, location.longitude)}
              </div>
              {location.altitude !== undefined && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  Altitude: <span style={{ color: '#fff' }}>{location.altitude.toFixed(2)} m</span>
                </div>
              )}
              {(location.country || location.state || location.city) && (
                <div style={{ fontSize: '12px', color: '#999', marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #444' }}>
                  {location.city && (
                    <div style={{ marginBottom: '4px' }}>
                      City: <span style={{ color: '#fff' }}>{location.city}</span>
                    </div>
                  )}
                  {location.state && (
                    <div style={{ marginBottom: '4px' }}>
                      State: <span style={{ color: '#fff' }}>{location.state}</span>
                    </div>
                  )}
                  {location.country && (
                    <div>
                      Country: <span style={{ color: '#fff' }}>{location.country}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* GPS Section */}
        {(metadata.latitude !== undefined || metadata.longitude !== undefined) && (
          <div>
            <h3 style={{ margin: '0 0 12px 0', color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              📍 GPS Coordinates
            </h3>
            <div
              style={{
                padding: '12px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                border: '1px solid #444',
              }}
            >
              {metadata.latitude !== undefined && metadata.longitude !== undefined && (
                <div style={{ fontSize: '14px', color: '#1976d2', marginBottom: '8px', fontWeight: '500' }}>
                  {formatGPS(metadata.latitude, metadata.longitude)}
                </div>
              )}
              {metadata.altitude !== undefined && (
                <div style={{ fontSize: '12px', color: '#999' }}>
                  Altitude: {metadata.altitude.toFixed(2)} m
                </div>
              )}
            </div>
          </div>
        )}

        {/* Camera Section */}
        {(metadata.camera_make || metadata.camera_model) && (
          <div>
            <h3 style={{ margin: '0 0 12px 0', color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              📷 Camera
            </h3>
            <div
              style={{
                padding: '12px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                border: '1px solid #444',
              }}
            >
              {metadata.camera_make && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  Make: <span style={{ color: '#fff' }}>{metadata.camera_make}</span>
                </div>
              )}
              {metadata.camera_model && (
                <div style={{ fontSize: '12px', color: '#999' }}>
                  Model: <span style={{ color: '#fff' }}>{metadata.camera_model}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* File Section */}
        {(metadata.file_size || metadata.file_type || metadata.mime_type || metadata.codec) && (
          <div>
            <h3 style={{ margin: '0 0 12px 0', color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              📁 File Info
            </h3>
            <div
              style={{
                padding: '12px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                border: '1px solid #444',
              }}
            >
              {metadata.file_size !== undefined && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  Size: <span style={{ color: '#fff' }}>{formatFileSize(metadata.file_size)}</span>
                </div>
              )}
              {metadata.file_type && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  Type: <span style={{ color: '#fff' }}>{metadata.file_type}</span>
                </div>
              )}
              {metadata.mime_type && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  MIME: <span style={{ color: '#fff' }}>{metadata.mime_type}</span>
                </div>
              )}
              {metadata.codec && (
                <div style={{ fontSize: '12px', color: '#999' }}>
                  Codec: <span style={{ color: '#fff' }}>{metadata.codec}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Temporal Section */}
        {(metadata.duration_seconds || metadata.frame_rate || metadata.create_date) && (
          <div>
            <h3 style={{ margin: '0 0 12px 0', color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              ⏱️ Temporal Info
            </h3>
            <div
              style={{
                padding: '12px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                border: '1px solid #444',
              }}
            >
              {metadata.duration_seconds !== undefined && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  Duration: <span style={{ color: '#fff' }}>{metadata.duration_seconds.toFixed(2)} s</span>
                </div>
              )}
              {metadata.frame_rate !== undefined && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  Frame Rate: <span style={{ color: '#fff' }}>{metadata.frame_rate.toFixed(2)} fps</span>
                </div>
              )}
              {metadata.create_date && (
                <div style={{ fontSize: '12px', color: '#999' }}>
                  Created: <span style={{ color: '#fff' }}>{formatDate(metadata.create_date)}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Image Section */}
        {(metadata.image_size || metadata.megapixels !== undefined || metadata.rotation !== undefined) && (
          <div>
            <h3 style={{ margin: '0 0 12px 0', color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              🖼️ Image Info
            </h3>
            <div
              style={{
                padding: '12px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                border: '1px solid #444',
              }}
            >
              {metadata.image_size && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  Size: <span style={{ color: '#fff' }}>{metadata.image_size}</span>
                </div>
              )}
              {metadata.megapixels !== undefined && (
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  Megapixels: <span style={{ color: '#fff' }}>{metadata.megapixels.toFixed(2)} MP</span>
                </div>
              )}
              {metadata.rotation !== undefined && (
                <div style={{ fontSize: '12px', color: '#999' }}>
                  Rotation: <span style={{ color: '#fff' }}>{metadata.rotation}°</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Bitrate Section */}
        {metadata.avg_bitrate && (
          <div>
            <h3 style={{ margin: '0 0 12px 0', color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              📊 Bitrate
            </h3>
            <div
              style={{
                padding: '12px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                border: '1px solid #444',
              }}
            >
              <div style={{ fontSize: '12px', color: '#999' }}>
                Average: <span style={{ color: '#fff' }}>{metadata.avg_bitrate}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
